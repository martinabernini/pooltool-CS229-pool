import math
import random

import numpy as np
import pooltool as pt
import pooltool.evolution as evo
import pooltool.constants as ptconst

# Table physical dimensions (meters)
TABLE_W = pt.Table.default().w   # 0.9906
TABLE_L = pt.Table.default().l   # 1.9812
BALL_R = 0.028575

# Scale factor: map table meters -> [0, 1000] pixel space expected by env.py
SCALE_X = 1000.0 / TABLE_W
SCALE_Y = 1000.0 / TABLE_L

# Max cue speed in m/s (corresponds to force=1.0 in normalized space)
MAX_V0 = 10.0

# Margin from table edge for random ball placement (meters)
MARGIN = BALL_R * 2


def _random_pos(existing: list[tuple[float, float]]) -> tuple[float, float]:
    """Return a random (x, y) on table that doesn't overlap existing balls."""
    min_dist = BALL_R * 2 + 0.005
    for _ in range(10000):
        x = random.uniform(MARGIN, TABLE_W - MARGIN)
        y = random.uniform(MARGIN, TABLE_L - MARGIN)
        if all(math.hypot(x - ex, y - ey) >= min_dist for ex, ey in existing):
            return x, y
    raise RuntimeError("Could not place ball without overlap after 10000 tries")


def _to_pixel(x: float, y: float) -> tuple[float, float]:
    return x * SCALE_X, y * SCALE_Y


class PooltoolGameState:
    """Drop-in replacement for gamestate.GameState using pooltool physics."""

    def __init__(self, num_balls: int, visualize: bool = False):
        self.num_balls = num_balls
        self.visualize = visualize  # pooltool headless; visualize flag kept for API compat
        self.collision_count = 0
        self._system: pt.System = self._build_system()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system(self) -> pt.System:
        placed: list[tuple[float, float]] = []

        # Cue ball placed in lower third of table
        cx = random.uniform(MARGIN, TABLE_W - MARGIN)
        cy = random.uniform(MARGIN, TABLE_L / 3)
        placed.append((cx, cy))

        balls: dict[str, pt.Ball] = {"cue": pt.Ball.create("cue", xy=(cx, cy))}

        for i in range(1, self.num_balls):
            x, y = _random_pos(placed)
            placed.append((x, y))
            balls[str(i)] = pt.Ball.create(str(i), xy=(x, y))

        table = pt.Table.default()
        cue_stick = pt.Cue(cue_ball_id="cue")
        return pt.System(cue=cue_stick, table=table, balls=balls)

    def _active_ball_ids(self) -> list[str]:
        """Return ids of balls not pocketed, cue ball first."""
        s = self._system
        all_ids = list(s.balls.keys())
        active = [bid for bid in all_ids if s.balls[bid].state.s != ptconst.pocketed]
        # put cue first
        if "cue" in active:
            active.remove("cue")
            active.insert(0, "cue")
        return active

    def _respot_cue(self):
        """Place cue ball back on table after it gets pocketed."""
        s = self._system
        others = [
            (s.balls[bid].state.rvw[0][0], s.balls[bid].state.rvw[0][1])
            for bid in s.balls
            if bid != "cue" and s.balls[bid].state.s != ptconst.pocketed
        ]
        cx = random.uniform(MARGIN, TABLE_W - MARGIN)
        cy = random.uniform(MARGIN, TABLE_L / 3)
        # avoid overlaps
        min_dist = BALL_R * 2 + 0.005
        for _ in range(10000):
            if all(math.hypot(cx - ox, cy - oy) >= min_dist for ox, oy in others):
                break
            cx = random.uniform(MARGIN, TABLE_W - MARGIN)
            cy = random.uniform(MARGIN, TABLE_L / 3)

        cue = s.balls["cue"]
        cue.state.rvw[0][:] = [cx, cy, BALL_R]
        cue.state.rvw[1][:] = 0.0
        cue.state.rvw[2][:] = 0.0
        cue.state.s = ptconst.stationary
        # clear from pockets
        for pocket in s.table.pockets.values():
            pocket.contains.discard("cue")

    # ------------------------------------------------------------------
    # Public interface (mirrors gamestate.GameState)
    # ------------------------------------------------------------------

    def return_ball_state(self) -> list[tuple[float, float]]:
        """Return (x, y) pixel-space positions, cue ball first."""
        s = self._system
        active = self._active_ball_ids()
        return [_to_pixel(s.balls[bid].state.rvw[0][0], s.balls[bid].state.rvw[0][1])
                for bid in active]

    def step(self, _game_unused, angle: float, force: float):
        """
        Execute one shot.

        Args:
            angle: normalized [0, 1] → mapped to phi in [0°, 360°]
            force: normalized [0, 1] → mapped to V0 in [0, MAX_V0] m/s

        Returns:
            (new_pos, balls_in, collision_count, done)
        """
        s = self._system

        phi = angle * 360.0
        V0 = force * MAX_V0

        # Reset simulation state but keep ball positions
        s.reset_history()
        for pocket in s.table.pockets.values():
            pocket.contains.clear()

        s.strike(V0=V0, phi=phi)
        evo.simulate(s, inplace=True)

        # Count ball-ball collisions and pocketed balls
        cue_pocketed = False
        balls_in = 0
        collision_count = 0
        for event in s.events:
            if event.event_type == "ball_ball" and "cue" in event.ids:
                collision_count += 1
            if event.event_type == "ball_pocket":
                bid = event.ids[0]
                if bid == "cue":
                    cue_pocketed = True
                else:
                    balls_in += 1

        if cue_pocketed:
            self._respot_cue()

        new_pos = self.return_ball_state()
        active_count = len(new_pos)

        # Pad to original ball_num with zeros (same as original gamestate)
        for _ in range(active_count, self.num_balls):
            new_pos.append((0, 0))

        done = 1 if active_count == 1 else 0
        self.collision_count = collision_count
        return new_pos, balls_in, collision_count, done
