/*
 * ml_obs.c — pack ml_obs_t from current game state, apply ml_action_t.
 *
 * This is the data-bridge layer between game.so internal entity state and
 * the wire format defined in ml_bridge.h.  Called from Bot_Think() when
 * the bot's zc.ml_enabled flag is set.
 */

#include "g_local.h"
#include "bot.h"
#include "ml_bridge.h"
#include <string.h>
#include <math.h>

/* Lithium server framenum source */
extern int sm_framenum;

/* Pack the bot's current state into an ml_obs_t structure. */
void ML_PackObs(edict_t *ent, ml_obs_t *obs)
{
	int       i, n;
	edict_t   *other;
	vec3_t    rel;
	zgcl_t    *zc = &ent->client->zc;

	memset(obs, 0, sizeof(*obs));
	obs->magic    = ML_OBS_MAGIC;
	obs->tick     = (uint32_t)level.framenum;
	obs->bot_slot = (uint32_t)(ent - g_edicts - 1);
	obs->yaw      = ent->s.angles[YAW];
	obs->pitch    = ent->s.angles[PITCH];

	/* ── self ──────────────────────────────────────────────── */
	VectorCopy(ent->s.origin,   obs->self.pos);
	VectorCopy(ent->velocity,   obs->self.vel);
	obs->self.health    = (float)ent->health;
	obs->self.armor     = (float)ent->client->pers.inventory[
		ITEM_INDEX(FindItem("Body Armor"))];
	obs->self.weapon_id = ent->client->pers.weapon
		? (float)ITEM_INDEX(ent->client->pers.weapon) : 0.0f;
	obs->self.ammo      = (float)(ent->client->pers.weapon && ent->client->pers.weapon->ammo
		? ent->client->pers.inventory[ITEM_INDEX(FindItem(ent->client->pers.weapon->ammo))]
		: 0);

	/* ── visible entities (other clients only for now) ─────── */
	n = 0;
	for (i = 1; i <= maxclients->value && n < ML_MAX_ENTITIES; i++)
	{
		other = &g_edicts[i];
		if (other == ent || !other->inuse || other->deadflag)
			continue;
		if (!other->client)
			continue;

		VectorSubtract(other->s.origin, ent->s.origin, rel);
		VectorCopy(rel,             obs->entities[n].rel_pos);
		VectorCopy(other->velocity, obs->entities[n].vel);
		obs->entities[n].health    = (float)other->health;
		obs->entities[n].is_enemy  = ctf->value
			? (other->client->resp.ctf_team != ent->client->resp.ctf_team ? 1.0f : 0.0f)
			: 1.0f;
		obs->entities[n].visible   = Bot_traceS(ent, other) ? 1.0f : 0.0f;
		n++;
	}
	obs->entity_count = (uint32_t)n;

	/* ── rays / hook zones ────────────────────────────────── */
	ML_FillRays(ent, obs);
	ML_FillHookZones(ent, obs);

	/* ── audio ────────────────────────────────────────────── */
	if (level.sound_entity && level.sound_entity_framenum >= level.framenum - 30)
	{
		VectorSubtract(level.sound_entity->s.origin, ent->s.origin, rel);
		float len = VectorLength(rel);
		if (len > 0.001f)
		{
			VectorScale(rel, 1.0f / len, obs->audio.sound_dir);
			obs->audio.sound_age = (float)(level.framenum - level.sound_entity_framenum);
			obs->audio.alert_level = 1.0f - (obs->audio.sound_age / 30.0f);
		}
	}

	/* ── reward components (cleared after send) ───────────── */
	obs->reward_damage_dealt   = zc->ml_reward_damage_dealt;
	obs->reward_damage_taken   = zc->ml_reward_damage_taken;
	obs->reward_kill           = zc->ml_reward_kill;
	obs->reward_death          = zc->ml_reward_death;
	obs->reward_item_pickup    = zc->ml_reward_item;
	obs->reward_hook_traversal = zc->ml_reward_hook;

	zc->ml_reward_damage_dealt = 0;
	zc->ml_reward_damage_taken = 0;
	zc->ml_reward_kill         = 0;
	zc->ml_reward_death        = 0;
	zc->ml_reward_item         = 0;
	zc->ml_reward_hook         = 0;

	obs->is_terminal = (ent->deadflag || level.intermissiontime > 0) ? 1 : 0;
}


/* Apply an action to the bot's edict — translates wire format into Q2
 * physics inputs and weapon triggers. */
void ML_ApplyAction(edict_t *ent, const ml_action_t *act)
{
	zgcl_t *zc = &ent->client->zc;
	float forward_speed = 320.0f;
	vec3_t forward, right;

	/* cache for any sub-tick logic */
	zc->ml_move_forward = act->move_forward;
	zc->ml_move_right   = act->move_right;
	zc->ml_look_yaw     = act->look_yaw;
	zc->ml_look_pitch   = act->look_pitch;
	zc->ml_jump         = act->jump;
	zc->ml_fire         = act->fire;
	zc->ml_hook         = act->hook;
	zc->ml_weapon       = act->weapon;

	/* ── look angles ──────────────────────────────────────── */
	ent->s.angles[YAW]   += act->look_yaw;
	ent->s.angles[PITCH] += act->look_pitch;
	if (ent->s.angles[PITCH] >  89.0f) ent->s.angles[PITCH] =  89.0f;
	if (ent->s.angles[PITCH] < -89.0f) ent->s.angles[PITCH] = -89.0f;
	ent->client->v_angle[YAW]   = ent->s.angles[YAW];
	ent->client->v_angle[PITCH] = ent->s.angles[PITCH];

	/* ── movement ─────────────────────────────────────────── */
	AngleVectors(ent->s.angles, forward, right, NULL);
	forward[2] = 0;
	right[2]   = 0;
	VectorNormalize(forward);
	VectorNormalize(right);

	if (ent->groundentity)
	{
		VectorScale(forward, act->move_forward * forward_speed, ent->velocity);
		VectorMA(ent->velocity, act->move_right * forward_speed, right, ent->velocity);

		/* preserve gravity */
		if (act->jump)
			ent->velocity[2] = VEL_BOT_JUMP;
	}
	else
	{
		/* air control: small adjustments only */
		ent->velocity[0] += forward[0] * act->move_forward * 30.0f;
		ent->velocity[1] += forward[1] * act->move_forward * 30.0f;
		ent->velocity[0] += right[0]   * act->move_right   * 30.0f;
		ent->velocity[1] += right[1]   * act->move_right   * 30.0f;
	}

	/* ── weapon select ────────────────────────────────────── */
	if (act->weapon > 0 && act->weapon < 10)
	{
		/* simple slot-based weapon selection — pick from the bot's BOP_PRIWEP
		 * style table.  For MVP we just no-op; integrating this properly
		 * means walking the inventory. */
	}

	/* ── fire ─────────────────────────────────────────────── */
	if (act->fire && ent->client->pers.weapon && ent->client->weaponstate == WEAPON_READY)
	{
		ent->client->buttons     |= BUTTON_ATTACK;
		ent->client->latched_buttons |= BUTTON_ATTACK;
	}
	else
	{
		ent->client->buttons     &= ~BUTTON_ATTACK;
	}

	/* ── hook (Lithium grapple) ───────────────────────────── */
	/* Hook integration arrives once map generator emits hook zones. */
	(void)act->hook;
}
