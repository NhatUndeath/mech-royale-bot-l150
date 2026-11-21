"""Survival-first mech bot.

Priorities, every decision:
1. Stay ahead of the zone (flow-field navigation on a grid built from the map).
2. Face whoever is shooting us (rear hits take x1.5), strafe while fighting.
3. Shoot back with lead; use laser / lightning / mortar when held.
4. Early and when safe: grab weapons and cubes, farm NPC drones and crates.
5. Retreat when losing a fight.
"""
import math
import os
import random
import sys
import time

import mechfight as mf
from nav import Nav

try:
    import params as P
except ImportError:
    P = None


def opt(name, default):
    return getattr(P, name, default) if P is not None else default


GATE = opt("GATE", False)            # only fight when attacked / close / finishing a weak target
RETREAT = opt("RETREAT", "v1")       # v1: hp<35% and losing; v2: hp<50% or crowded
COVER = opt("COVER", False)          # hold a spot near walls inside the next zone
FARM = opt("FARM", True)             # shoot NPC drones and crates when quiet
DODGE = opt("DODGE", False)          # sidestep lasers aimed at us
BREAK_LOS = opt("BREAK_LOS", False)  # retreat toward a spot the attacker cannot see
EDGE = opt("EDGE", 0.5)
LOOT_EARLY = opt("LOOT_EARLY", 150)
BERSERK = opt("BERSERK", False)
CONTEXT = opt("CONTEXT", False)      # detect a free-for-all (others die without us) and hide until the end game
HIDE_UNTIL = opt("HIDE_UNTIL", 4)
PEEK = opt("PEEK", False)            # duck behind cover while the gun reloads, then peek back out
PEEK_R = opt("PEEK_R", 14)           # how far to look for cover (m)
PEEK_FOES = opt("PEEK_FOES", 4)      # hide from this many of the nearest visible mechs
BACK_W = opt("BACK_W", 0)            # target bonus (m) for a mech whose back is turned to us (rear hits x1.5)
LOWHP_W = opt("LOWHP_W", 40)         # target bonus (m) per missing HP fraction
HOLD = opt("HOLD", True)            # never push toward the target while 2+ mechs are aiming at us

UPGRADE_PREF = ["armor", "plating", "nanites", "gun_rate", "gun_damage", "laser_damage", "laser_charge",
                "servos", "gun_mag", "gun_velocity", "sensors", "gyros", "reactor", "capacitor", "jets"]
WEAPON_PREF = opt("WEAPON_PREF", {"laser": 3, "lightning": 2, "mortar": 1, "gun": 0})
CLOSE_DIST = opt("CLOSE_DIST", 35)   # with lightning, close in to this distance
CLOSE_W = opt("CLOSE_W", 0.6)        # how hard to push in while strafing


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def norm2(x, y):
    l = math.hypot(x, y)
    return (x / l, y / l) if l > 1e-6 else (0.0, 0.0)


class Bot:
    name = "survivor"

    def __init__(self, hello):
        cfg = hello["config"]
        self.cfg = cfg
        self.rng = random.Random(hello.get("seed", 0))
        mdir = hello.get("map", {}).get("dir") or "/arena"
        if not os.path.isdir(mdir):
            for alt in ("/arena", os.path.join(os.path.dirname(__file__), "arena")):
                if os.path.isdir(alt):
                    mdir = alt
                    break
        self.nav = Nav(mdir)
        self.goal = None
        self.goal_key = None
        self.last_pos = None
        self.stuck_t = 0.0
        self.unstick_until = -1.0
        self.unstick_dir = (1.0, 0.0)
        self.threat = {}          # mech id -> (time, bearing)
        self.last_hit_t = -99.0
        self.last_hit_bearing = None
        self.strafe_sign = 1
        self.strafe_until = 0.0
        self.scan_yaw = 0.0
        self.sound_look = None    # (until, bearing)
        self.prev_t = 0.0
        self.laser_hold = 0.0
        self.cur_target = None
        self.peek_from = None     # (x, y, t) where we last shot from before ducking
        self.peek_until = -1.0
        self.last_tpos = None
        self.cover_cache = (-9.0, None, None)
        self.ffa = False          # evidence that other mechs fight each other
        self.hiding = False
        self.prev_alive = None
        self.my_kill_t = -99.0
        self.hp_seen = {}         # mech id -> last hp we saw
        self.my_hit_t = {}        # mech id -> last time we hit it
        self.ffa_score = 0
        self.ffa_deaths = 0
        self.ffa_hits = 0
        self.dmg_from = {}        # mech id -> list of (time, amount)

    # ------------------------------------------------------------------ helpers
    def weapon_slots(self, me):
        out = {}
        for i, w in enumerate(me.get("weapons") or []):
            if w:
                out[w["kind"]] = (i, w)
        return out

    def zone_goal(self, obs, me):
        z = obs["zone"]
        nc, nr = z["next_center"], z["next_radius"]
        key = (round(nc[0], 1), round(nc[1], 1), round(nr, 1))
        x, y = me["pos"][0], me["pos"][1]
        if key != self.goal_key or self.goal is None:
            d = math.dist((x, y), nc)
            r = EDGE * nr
            if d <= r:
                g = (x, y)
            else:
                ux, uy = norm2(x - nc[0], y - nc[1])
                g = (nc[0] + ux * r, nc[1] + uy * r)
            if (COVER or self.hiding) and self.nav.ready and nr > 20:
                g = self.nav.cover_spot(nc, (EDGE + 0.05) * nr, g, self.rng) or g
            if self.nav.ready:
                i, j = self.nav.nearest_free(g[0], g[1], 80, main=True)
                g = self.nav.center(i, j)
            self.goal, self.goal_key = g, key
        self.nav.request(self.goal)
        return self.goal

    def must_travel(self, obs, me):
        """True when the zone leaves no time for detours."""
        z = obs["zone"]
        t = obs["t"]
        x, y = me["pos"][0], me["pos"][1]
        if not me.get("in_zone", True):
            return True
        need = max(0.0, math.dist((x, y), z["next_center"]) - 0.75 * z["next_radius"])
        speed = me["stats"].get("walk_speed", 11.7) * 0.6
        avail = max(0.0, z.get("shrink_end", t) - t) * speed
        return need > avail - 40

    def lead(self, me, tgt, speed):
        mx, my, mz = me["pos"]
        mz += me["stats"].get("muzzle_height", 7)
        tx, ty, tz = tgt["pos"]
        h = 4.5 if tgt["kind"] == "mech" else 1.75
        if tgt["kind"] == "mech":
            h *= 1 + 0.05 * (tgt.get("cubes") or 0)
        tz += h
        vx, vy, vz = tgt.get("vel", [0, 0, 0])
        t = 0.0
        for _ in range(3):
            px, py, pz = tx + vx * t, ty + vy * t, tz + (vz * t if abs(vz) < 3 else 0)
            d = math.dist((mx, my, mz), (px, py, pz))
            t = d / speed + 0.05
        yaw = math.atan2(py - my, px - mx)
        pitch = math.atan2(pz - mz, math.hypot(px - mx, py - my))
        return yaw, pitch, math.hypot(px - mx, py - my)

    def mortar_aim(self, me, tgt, wstats):
        mx, my, mz = me["pos"]
        mz += me["stats"].get("height", 9) + 0.5
        tx, ty, tz = tgt["pos"]
        v, g = wstats.get("speed", 72), wstats.get("gravity", 22)
        x = math.hypot(tx - mx, ty - my)
        yv = tz + 1 - mz
        disc = v ** 4 - g * (g * x * x + 2 * yv * v * v)
        if disc < 0 or x < 1:
            return None
        ang = math.atan2(v * v - math.sqrt(disc), g * x)
        tof = x / (v * math.cos(ang))
        vx, vy, _ = tgt.get("vel", [0, 0, 0])
        tx2, ty2 = tx + vx * tof, ty + vy * tof
        x2 = math.hypot(tx2 - mx, ty2 - my)
        disc = v ** 4 - g * (g * x2 * x2 + 2 * yv * v * v)
        if disc < 0:
            return None
        ang = math.atan2(v * v - math.sqrt(disc), g * x2)
        return math.atan2(ty2 - my, tx2 - mx), ang

    def threat_value(self, m, x, y, t):
        """How urgent it is to shoot m: its damage rate on us divided by the time we need to kill it."""
        d = max(1.0, math.dist((x, y), (m["pos"][0], m["pos"][1])))
        b = math.atan2(y - m["pos"][1], x - m["pos"][0])
        facing = abs(wrap(m["aim"][0] - b)) < 0.35
        ws = m.get("weapons") or ["gun"]
        dps = 0.0
        if d < 230:
            dps += 150 if facing else 40
        if "laser" in ws and d < 280:
            dps += (260 if m.get("charging") and facing else 110)
        if "lightning" in ws and d < 50:
            dps += 200
        if "mortar" in ws and 20 < d < 230:
            dps += 70
        recent = sum(a for (tt, a) in self.dmg_from.get(m["id"], []) if t - tt < 3)
        dps += recent / 3.0 * 1.5
        ttk = max(0.5, m.get("hp", 1400) / 160.0) * (1 + d / 250)
        return dps / ttk

    def hide_dir(self, x, y, z, enemy):
        """Direction to the nearest reachable point (within ~25 m) the enemy has no line of sight to."""
        ex, ey, ez = enemy
        best = None
        for k in range(16):
            a = k * math.pi / 8
            ca, sa = math.cos(a), math.sin(a)
            for r in (8, 14, 20, 26):
                px, py = x + ca * r, y + sa * r
                if not self.nav.free(px, py):
                    break
                if not self.nav.los((ex, ey, ez + 8), (px, py, z + 4.5)):
                    score = r - 6 * (ca * (x - ex) + sa * (y - ey)) / max(1.0, math.hypot(x - ex, y - ey))
                    if best is None or score < best[0]:
                        best = (score, ca, sa)
                    break
        return (best[1], best[2]) if best else None

    def cover_from(self, x, y, z, foes, t):
        """Direction to the closest reachable point no foe can see (None if none within PEEK_R)."""
        ct, cpos, cdir = self.cover_cache
        if t - ct < 0.3 and cpos is not None and math.dist(cpos, (x, y)) < 2:
            return cdir
        best = None
        if foes:
            # already hidden?
            if all(not self.nav.los((fx, fy, fz + 7), (x, y, z + 4.5)) for fx, fy, fz in foes):
                self.cover_cache = (t, (x, y), (0.0, 0.0))
                return (0.0, 0.0)
            rs = [r for r in (3, 6, 9, 12, 15, 18, 22, 26, 30) if r <= PEEK_R]
            for k in range(16):
                a = k * math.pi / 8
                ca, sa = math.cos(a), math.sin(a)
                for r in rs:
                    px, py = x + ca * r, y + sa * r
                    if not self.nav.free(px, py):
                        break
                    pz = self.nav.ground_h(px, py)
                    if all(not self.nav.los((fx, fy, fz + 7), (px, py, pz + 5)) for fx, fy, fz in foes):
                        if best is None or r < best[0]:
                            best = (r, ca, sa)
                        break
        d = (best[1], best[2]) if best else None
        self.cover_cache = (t, (x, y), d)
        return d

    def aligned(self, me, yaw, pitch, dist, tol_r=2.5):
        ay, ap = me["aim"]
        tol = math.atan2(tol_r, max(dist, 1.0)) + 0.01
        return abs(wrap(ay - yaw)) < tol and abs(ap - pitch) < tol + 0.03

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t_start = time.perf_counter()
        me = obs["self"]
        t = obs["t"]
        dt = max(0.05, t - self.prev_t) if self.prev_t else 0.1
        self.prev_t = t
        x, y, z = me["pos"]
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        act = {"tick": obs["tick"]}
        dbg = []

        # upgrades
        off = obs.get("offer")
        if off:
            ids = [c["id"] for c in off["choices"]]
            best = min(range(len(ids)), key=lambda i: UPGRADE_PREF.index(ids[i]) if ids[i] in UPGRADE_PREF else 99)
            act["pick"] = best

        # free-for-all detection: in practice the house squad never fights itself,
        # so mechs dying or losing HP without us means a real free-for-all
        for ev in obs.get("events") or []:
            if ev["type"] == "kill":
                self.my_kill_t = t
            elif ev["type"] == "hit":
                self.my_hit_t[ev.get("target")] = t
        alive = obs.get("alive")
        if CONTEXT and alive is not None:
            if self.prev_alive is not None and alive < self.prev_alive and t - self.my_kill_t > 1.5:
                self.ffa_deaths += self.prev_alive - alive
            for m in obs.get("mechs") or []:
                if m["kind"] != "mech" or "hp" not in m:
                    continue
                old = self.hp_seen.get(m["id"])
                if (old is not None and t - old[0] < 0.35 and m["hp"] < old[1] - 30
                        and t - self.my_hit_t.get(m["id"], -99) > 1.0):
                    self.ffa_hits += 1
                self.hp_seen[m["id"]] = (t, m["hp"])
            self.ffa_score = 2 * self.ffa_deaths + self.ffa_hits
            if self.ffa_deaths >= 3 or (self.ffa_deaths >= 2 and self.ffa_hits >= 3):
                self.ffa = True
        self.prev_alive = alive
        self.hiding = CONTEXT and self.ffa and (alive or 16) > HIDE_UNTIL

        # threats from events and sounds
        for ev in obs.get("events") or []:
            if ev["type"] == "damaged" and ev.get("by") is not None and ev.get("bearing") is not None:
                self.threat[ev["by"]] = (t, ev["bearing"])
                self.last_hit_t, self.last_hit_bearing = t, ev["bearing"]
                self.dmg_from.setdefault(ev["by"], []).append((t, ev.get("amount", 0)))
        for s in obs.get("sounds") or []:
            if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet") and s["dist"] == "near":
                if self.sound_look is None or self.sound_look[0] < t:
                    self.sound_look = (t + 1.0, s["bearing"])

        mechs = [m for m in obs.get("mechs") or [] if m["kind"] == "mech"]
        npcs = [m for m in obs.get("mechs") or [] if m["kind"] == "npc"]
        slots = self.weapon_slots(me)

        def d2(p):
            return math.dist((x, y), (p[0], p[1]))

        # ---------------- target choice
        target = None
        if mechs:
            def score(m):
                d = d2(m["pos"])
                s = d
                if m["id"] in self.threat and t - self.threat[m["id"]][0] < 4:
                    s -= 60
                s -= LOWHP_W * (1 - (m.get("hp", 1400) / max(1, m.get("max_hp", 1400))))
                if BACK_W and m.get("aim"):
                    to_us = math.atan2(y - m["pos"][1], x - m["pos"][0])
                    if abs(wrap(m["aim"][0] - to_us)) > math.pi / 2:
                        s -= BACK_W
                return s
            if BERSERK:
                target = max(mechs, key=lambda m: self.threat_value(m, x, y, t))
                # stick with the current target unless another is clearly better
                cur = next((m for m in mechs if m["id"] == self.cur_target), None)
                if cur is not None and self.threat_value(cur, x, y, t) > 0.75 * self.threat_value(target, x, y, t):
                    target = cur
                self.cur_target = target["id"]
            else:
                target = min(mechs, key=score)
        tdist = d2(target["pos"]) if target else 1e9
        if target is not None:
            self.last_tpos = (target["pos"][0], target["pos"][1], t)

        travel = self.must_travel(obs, me)
        goal = self.zone_goal(obs, me)
        mode = "zone"
        move_to = goal

        # ---------------- farming / looting when safe
        nearest_mech = min((d2(m["pos"]) for m in mechs), default=1e9)
        safe = not travel and nearest_mech > 90 and hp_frac > 0.45
        farm_target = None
        if safe:
            empty = any(w is None for w in me["weapons"])
            picks = []
            for p in obs.get("pickups") or []:
                if p["kind"] in slots or p["kind"] == "gun":
                    continue
                gain = WEAPON_PREF.get(p["kind"], 0)
                if not empty:
                    worst = min((WEAPON_PREF.get(k, 0) for k in slots if k != "gun"), default=99)
                    if gain <= worst:
                        continue
                if abs(p["pos"][2] - self.nav.ground_h(p["pos"][0], p["pos"][1])) > 3:
                    continue
                if d2(p["pos"]) < (LOOT_EARLY if t < 200 else 70):
                    picks.append(p)
            cubes = [c for c in obs.get("cubes") or [] if d2(c["pos"]) < 60 and abs(c["pos"][2] - self.nav.ground_h(c["pos"][0], c["pos"][1])) < 3]
            if picks:
                p = min(picks, key=lambda p: d2(p["pos"]))
                move_to, mode = (p["pos"][0], p["pos"][1]), "loot"
                if d2(p["pos"]) < self.cfg["loot"]["pickup_radius"] - 0.3:
                    act["take"] = p["id"]
                    if not empty:
                        worst_slot = min((s for k, (s, w) in slots.items() if k != "gun"),
                                         key=lambda s: WEAPON_PREF.get(me["weapons"][s]["kind"], 0))
                        act["drop"] = worst_slot
            elif cubes:
                c = min(cubes, key=lambda c: d2(c["pos"]))
                move_to, mode = (c["pos"][0], c["pos"][1]), "cube"
            # farm NPC drones (2 cubes) and crates (1 cube + maybe a weapon) while it is quiet
            if FARM and t < 260 and hp_frac > 0.6 and not mechs:
                npc_c = [n for n in npcs if d2(n["pos"]) < 70]
                crates = [c for c in obs.get("crates") or [] if d2(c["pos"]) < 70
                          and abs(c["pos"][2] - self.nav.ground_h(c["pos"][0], c["pos"][1])) < 3]
                if npc_c:
                    farm_target = dict(min(npc_c, key=lambda n: d2(n["pos"])))
                elif crates and mode == "zone":
                    c = min(crates, key=lambda c: d2(c["pos"]))
                    farm_target = {"kind": "crate", "pos": c["pos"], "vel": [0, 0, 0], "id": c["id"]}
        # supply drop: only if very close and nobody around
        if safe and mode == "zone" and not mechs:
            for dr in obs.get("drops") or []:
                if dr["lands_in"] <= 3 and d2(dr["pos"]) < 70 and (dr.get("holder") in (None, me["id"])):
                    move_to, mode = (dr["pos"][0], dr["pos"][1]), "drop"

        # ---------------- movement
        gx, gy = move_to
        dgoal = math.dist((x, y), (gx, gy))
        mv = (0.0, 0.0)
        if mode == "zone":
            if dgoal > 6:
                s = self.nav.steer(x, y, (gx, gy))
                mv = norm2(*(s if s else (gx - x, gy - y)))
        else:
            if dgoal > 1.0:
                s = None
                if dgoal > 15:
                    s = None  # direct is fine for short hops; walls slide us
                mv = norm2(gx - x, gy - y)
        boost = not me.get("in_zone", True) or travel and dgoal > 60
        if PEEK and target is None and self.peek_from is not None and not travel and me.get("in_zone", True):
            px_, py_, pt_ = self.peek_from
            if t < self.peek_until:
                mv, mode = (0.0, 0.0), "duck"
            elif t - pt_ < 6 and math.dist((x, y), (px_, py_)) > 2:
                mv, mode = norm2(px_ - x, py_ - y), "peek"

        # combat movement: strafe / retreat
        engage = False
        if target is not None:
            their = target.get("hp", 1400) / max(1, target.get("max_hp", 1400))
            attacked = target["id"] in self.threat and t - self.threat[target["id"]][0] < 5
            engage = (not (GATE or self.hiding)) or attacked or tdist < 60 or (their < 0.35 and hp_frac > 0.6 and tdist < 150)
            if not engage and tdist < 180:
                # stay out of it: move away from them, keep facing them
                ax, ay = norm2(x - target["pos"][0], y - target["pos"][1])
                w = (180 - tdist) / 180
                mv = norm2(mv[0] + ax * w * 1.5, mv[1] + ay * w * 1.5)
                mode = "avoid"
        fighting = engage and tdist < 160
        if fighting:
            ty_ = math.atan2(target["pos"][1] - y, target["pos"][0] - x)
            px, py = -math.sin(ty_), math.cos(ty_)
            if t > self.strafe_until:
                self.strafe_sign = -self.strafe_sign if self.rng.random() < 0.7 else self.strafe_sign
                self.strafe_until = t + self.rng.uniform(0.6, 1.6)
            sx, sy = px * self.strafe_sign, py * self.strafe_sign
            their = target.get("hp", 1400) / max(1, target.get("max_hp", 1400))
            crowd = sum(1 for m in mechs if d2(m["pos"]) < 120)
            if RETREAT == "v2":
                retreat = (hp_frac < 0.5 and their > hp_frac - 0.05) or (crowd >= 2 and hp_frac < 0.8)
            else:
                retreat = hp_frac < 0.35 and their > hp_frac
            if retreat:
                ax, ay = -math.cos(ty_), -math.sin(ty_)
                if BREAK_LOS and self.nav.ready:
                    hide = self.hide_dir(x, y, z, target["pos"])
                    if hide:
                        ax, ay = hide
                mv = norm2(ax * 0.7 + sx * 0.5 + mv[0] * 0.6, ay * 0.7 + sy * 0.5 + mv[1] * 0.6)
                boost = me["energy"] > 25
                mode = "retreat"
            else:
                want_close = "lightning" in slots and tdist > CLOSE_DIST
                fx, fy = (math.cos(ty_) * CLOSE_W / 0.6, math.sin(ty_) * CLOSE_W / 0.6) if want_close else (0.0, 0.0)
                mv = norm2(sx + mv[0] * 0.5 + fx * 0.6, sy + mv[1] * 0.5 + fy * 0.6)
                mode = "fight"
                if HOLD:
                    aiming = 0
                    for m in mechs:
                        if m.get("aim") and d2(m["pos"]) < 160:
                            to_us = math.atan2(y - m["pos"][1], x - m["pos"][0])
                            if abs(wrap(m["aim"][0] - to_us)) < 0.35:
                                aiming += 1
                    if aiming >= 2:
                        ux, uy = math.cos(ty_), math.sin(ty_)
                        fwd = mv[0] * ux + mv[1] * uy
                        if fwd > -0.3:   # drop any push toward the target, drift slightly back
                            mv = norm2(mv[0] - (fwd + 0.3) * ux, mv[1] - (fwd + 0.3) * uy)
                        mode = "hold"
            # do not strafe into walls
            if self.nav.ready and not self.nav.free(x + mv[0] * 4, y + mv[1] * 4):
                self.strafe_sign = -self.strafe_sign
                self.strafe_until = t + 1.0
            gun_ = slots.get("gun")
            rl = gun_[1].get("reload", 0) if gun_ else 0
            if PEEK and not retreat and self.nav.ready and gun_ and (rl > 0 or gun_[1].get("ammo", 1) == 0):
                foes = [m["pos"] for m in sorted(mechs, key=lambda m: d2(m["pos"]))[:PEEK_FOES] if d2(m["pos"]) < 160]
                h = self.cover_from(x, y, z, foes, t)
                if h is not None:
                    if self.peek_until < t:
                        self.peek_from = (x, y, t)
                    self.peek_until = t + max(rl, 0.3)
                    mv = norm2(*h) if h != (0.0, 0.0) else (0.0, 0.0)
                    boost = h != (0.0, 0.0) and me["energy"] > 30
                    mode = "duck"
        # sidestep a laser being charged at us
        if DODGE:
            for m in mechs:
                if m.get("charging"):
                    b = math.atan2(y - m["pos"][1], x - m["pos"][0])
                    if abs(wrap(m["aim"][0] - b)) < 0.15:
                        px, py = -math.sin(b), math.cos(b)
                        sgn = 1 if (int(t * 2) % 2 == 0) else -1
                        mv = norm2(px * sgn * 2 + mv[0], py * sgn * 2 + mv[1])
                        boost = me["energy"] > 10
                        mode = "dodge"
                        break
        # NPC avoidance when not farming them
        if farm_target is None or farm_target.get("kind") != "npc":
            for n in npcs:
                dn = d2(n["pos"])
                if dn < 70:
                    ax, ay = norm2(x - n["pos"][0], y - n["pos"][1])
                    w = 1.5 * (70 - dn) / 70
                    mv = norm2(mv[0] + ax * w, mv[1] + ay * w)

        # stuck handling
        speed = math.hypot(me["vel"][0], me["vel"][1])
        if mv != (0.0, 0.0) and speed < 1.5 and not fighting:
            self.stuck_t += dt
        else:
            self.stuck_t = max(0.0, self.stuck_t - dt)
        if self.stuck_t > 0.8 and t > self.unstick_until:
            ang = math.atan2(mv[1], mv[0]) + self.rng.choice((-1, 1)) * self.rng.uniform(0.8, 1.8)
            self.unstick_dir = (math.cos(ang), math.sin(ang))
            self.unstick_until = t + 0.8
            self.stuck_t = 0.0
        if t < self.unstick_until:
            mv = self.unstick_dir
            if me.get("grounded") and me["energy"] > 30:
                act["jump"] = True
            boost = me["energy"] > 20
        elif self.stuck_t > 0.4 and me.get("grounded"):
            act["jump"] = True
        if not me.get("grounded") and me["energy"] > 15 and self.stuck_t > 0.3:
            boost = True

        act["move"] = [round(mv[0], 3), round(mv[1], 3)]
        act["boost"] = bool(boost)

        # ---------------- aiming and firing
        fire = [False, False]
        shoot_at = None
        if target is not None and engage and tdist < 230:
            shoot_at = target
        elif target is not None:
            yaw, pitch, hd = self.lead(me, target, 252)
            act["aim"] = [yaw, 0.0]
        elif farm_target is not None:
            shoot_at = farm_target

        if shoot_at is not None:
            gun = slots.get("gun")
            gspeed = gun[1]["stats"]["speed"] if gun else 252
            yaw, pitch, hd = self.lead(me, shoot_at, gspeed)
            act["aim"] = [yaw, max(-0.69, min(1.29, pitch))]
            aligned = self.aligned(me, yaw, pitch, hd, 2.0 if shoot_at.get("kind") == "mech" else 1.4)
            eye = (x, y, z + me["stats"].get("muzzle_height", 7))
            tp = (shoot_at["pos"][0], shoot_at["pos"][1], shoot_at["pos"][2] + 3)
            clear = self.nav.los(eye, tp) if hd < 230 else False
            dc = self.__dict__.setdefault("dbg_counts", {})
            dc["tgt"] = dc.get("tgt", 0) + 1
            dc["aligned"] = dc.get("aligned", 0) + bool(aligned)
            dc["clear"] = dc.get("clear", 0) + bool(clear)
            dc["inrange"] = dc.get("inrange", 0) + bool(hd < 230)
            dc["reloading"] = dc.get("reloading", 0) + bool(gun and gun[1].get("reload", 0) > 0)
            dc["ok"] = dc.get("ok", 0) + bool(aligned and clear and hd < 230)
            if gun and aligned and clear and hd < gun[1]["stats"].get("range", 230):
                if gun[1].get("reload", 0) == 0 and gun[1].get("ammo", 1) > 0:
                    fire[gun[0]] = True
            lz = slots.get("laser")
            if lz:
                i, w = lz
                full = w["stats"].get("charge", 0.75)
                ch = w.get("charge", 0) or 0
                if shoot_at.get("kind") == "mech" and hd < w["stats"].get("range", 280) and w.get("cooldown", 0) <= 0.05:
                    if ch < full or not (aligned and clear):
                        fire[i] = ch < full + 1.5   # keep charging, release before auto-fire
                    else:
                        fire[i] = False             # release = fire
            lt = slots.get("lightning")
            if lt:
                i, w = lt
                if hd < w["stats"].get("range", 45) * 0.95 and abs(wrap(me["aim"][0] - yaw)) < math.radians(w["stats"].get("cone_deg", 30)) * 0.45 \
                        and me["energy"] > 10 and clear:
                    fire[i] = True
            mo = slots.get("mortar")
            if mo and shoot_at.get("kind") == "mech" and 25 < hd < 220:
                i, w = mo
                sol = self.mortar_aim(me, shoot_at, w["stats"])
                if sol:
                    if w.get("cooldown", 0) <= 0.1:
                        act["aim"] = [sol[0], max(-0.69, min(1.29, sol[1]))]
                        if abs(wrap(me["aim"][0] - sol[0])) < 0.03 and abs(me["aim"][1] - sol[1]) < 0.03:
                            fire[i] = True
        elif target is None:
            # no target: look at threats, then sounds, else scan along travel
            look = None
            if self.last_hit_bearing is not None and t - self.last_hit_t < 3:
                look = self.last_hit_bearing
            elif self.sound_look and self.sound_look[0] > t:
                look = self.sound_look[1]
            if look is None and PEEK and self.last_tpos is not None and t - self.last_tpos[2] < 6:
                look = math.atan2(self.last_tpos[1] - y, self.last_tpos[0] - x)
            if look is None:
                base = math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else self.scan_yaw
                self.scan_yaw = wrap(self.scan_yaw + 1.6 * dt)
                look = wrap(base + 1.0 * math.sin(t * 0.9)) if mv != (0.0, 0.0) else self.scan_yaw
            act["aim"] = [look, 0.0]
            gun = slots.get("gun")
            if gun and gun[1].get("reload", 0) == 0 and gun[1].get("ammo", 99) < gun[1]["stats"].get("mag", 12) * 0.5:
                act["reload"] = True
        # never keep a laser charging without a target
        lz = slots.get("laser")
        if lz and shoot_at is None and (lz[1].get("charge") or 0) > 0:
            fire[lz[0]] = False
        act["fire"] = fire

        self.nav.work(max(0.0, 0.010 - (time.perf_counter() - t_start)))
        # debug record (only our team can download it): enough to rebuild why we die
        w1 = me["weapons"][1]["kind"] if me["weapons"][1] else "-"
        mf.draw.label(f"{'H' if self.hiding else ('F' if self.ffa else 'S')} {mode} hp{me['hp']:.0f}/{me['max_hp']:.0f} e{me['energy']:.0f} {w1} "
                      f"T{target['id'] if target else '-'}@{tdist if target else 0:.0f} n{len(mechs)} st{self.stuck_t:.1f}")
        if self.goal:
            mf.draw.line([x, y, z + 2], [self.goal[0], self.goal[1], z + 2], "#40c0ff")
        if shoot_at is not None:
            mf.draw.line([x, y, z + 7], [shoot_at["pos"][0], shoot_at["pos"][1], shoot_at["pos"][2] + 4], "#ff5040")
        dmg = [e for e in obs.get("events") or [] if e["type"] == "damaged"]
        if dmg:
            mf.draw.text(" ".join(f"{e['weapon']}:{e['amount']:.0f}{'R' if e.get('rear') else ''}<{e.get('by')}" for e in dmg[:6]))
        return act


if __name__ == "__main__":
    mf.run(Bot)
