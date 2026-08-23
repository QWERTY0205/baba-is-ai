from dataclasses import dataclass
from functools import partial
from typing import Iterable
import numpy as np

from baba.grid import BabaIsYouGrid, BabaIsYouEnv, put_rule, place_rule
from baba.play import play
from baba.world_object import FBall, Baba, make_obj, RuleColor, RuleIs, RuleObject, RuleProperty, Wall
from baba import make, register



def put_obj(env, obj, pos):
    """
    Put an object at a position in the grid.
    Args:
        env: environment
        obj: object to put, tuple (color, name) or string name
        pos: position to put the object, (i, j)
    """
    if isinstance(obj, tuple):
        color, obj = obj
    else:
        color = None
    obj = make_obj(obj, color=color) if isinstance(obj, str) else obj
    env.put_obj(obj, *pos)


def place_obj(env, obj, top=None, size=None):
    """
    Place an object at a random empty position in the grid.
    Args:
        env: environment
        obj: object to place, tuple (color, name) or string name
        top: top left position of the area to place the object
        size: size of the area to place the object
    """
    if isinstance(obj, tuple):
        color, obj = obj
    else:
        color = None
    obj = make_obj(obj, color=color) if isinstance(obj, str) else obj
    pos = env.place_obj(obj, top=top, size=size)
    return pos


def break_rule(env, rule, new_pos={}, block_idx=None):
    """
    Break a rule by moving one of its blocks to a new position.
    Args:
        rule: rule to break
        new_pos: new position for the block
        block_idx: index of the block to move
    """
    rule_pos = env.init_rules[rule]

    if isinstance(new_pos, dict):
        new_pos = env.place_obj(None, **new_pos)  # pos constraints the new_pos for the rule block

    if block_idx is None:
        # pick one block of the rule and displace it
        block_idx = np.random.choice(len(rule_pos))
    elif isinstance(block_idx, Iterable):
        block_idx = block_idx[np.random.choice(len(block_idx))]

    p = rule_pos[block_idx]
    env.change_obj_pos(p, new_pos)
    # env.solution[rule] = {
    #     "push": (new_pos, p)
    # }


NAMES = ["ball", "key", "door"]
COLORS = ["red", "blue", "green"]
OBJECTS = [
    (color, name) for color in COLORS for name in NAMES
]
ACTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}

@register("env/you_win")
class YouWinEnv(BabaIsYouEnv):
    def __init__(self, width=6, height=6, fixed_you=False, **kwargs):
        self.fixed_you = fixed_you
        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        # randomly sample the parameters
        indices = np.random.choice(len(OBJECTS), size=4, replace=False)
        win_obj, you_obj, distractor1, distractor2 = [OBJECTS[i] for i in indices]

        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        put_obj(self, Wall(), (1, 3))
        put_obj(self, Wall(), (1, 4))
        put_obj(self, Wall(), (4, 3))
        put_obj(self, Wall(), (4, 4))

        if not self.fixed_you:
            # randomly sample the positions
            all_pos = [(2, 3), (3, 3), (2, 4), (3, 4)]
            you_obj_pos = all_pos.pop(np.random.choice(4))
        else:
            you_obj_pos = (2, 3)
            all_pos = [(2, 4), (3, 3), (3, 4)]

        # make sure the win object is one cell away from the you object
        # e.g. if you object is at (2, 3), win object can be at (2, 4) or (3, 3)
        while True:
            idx = np.random.choice(3)
            win_obj_pos = all_pos[idx]
            if abs(win_obj_pos[0] - you_obj_pos[0]) + abs(win_obj_pos[1] - you_obj_pos[1]) == 1:
                break
        all_pos.pop(idx)
        distractor1_pos = all_pos.pop(np.random.choice(2))
        distractor2_pos = all_pos[0]

        put_rule(self, you_obj, "you", positions=(1, 1))
        put_rule(self, win_obj, "win", positions=(1, 2))
        put_obj(self, you_obj, you_obj_pos)
        put_obj(self, win_obj, win_obj_pos)
        put_obj(self, distractor1, distractor1_pos)
        put_obj(self, distractor2, distractor2_pos)

        # action to move the you object to the win object
        target_action = None
        for action, (dx, dy) in ACTIONS.items():
            if (you_obj_pos[0] + dx, you_obj_pos[1] + dy) == win_obj_pos:
                target_action = action
                break
        self.target_action = target_action

        self.objects = {
            you_obj: you_obj_pos,
            win_obj: win_obj_pos,
            distractor1: distractor1_pos,
            distractor2: distractor2_pos
        }

        self.active_rules = [
            f"{you_obj[0]} {you_obj[1]} is you",
            f"{win_obj[0]} {win_obj[1]} is win"
        ]

        self.target_plan = f"goto[{win_obj}]"


@register("env/you_win-fixed_you")
class YouWinFixedYouEnv(YouWinEnv):
    def __init__(self, **kwargs):
        super().__init__(fixed_you=True, **kwargs)


@register("env/make_win-distr_obj_rule")
class MakeWinEnv(BabaIsYouEnv):
    def __init__(
            self,
            width=8,
            height=8,
            color_in_rule=False,
            break_win_rule=True,
            distractor_obj=True,
            distractor_rule_block=True,
            irrelevant_rule_distractor=False,
            distractor_win_rule=False,
            win_obj_set=None,
            **kwargs
        ):
        self.color_in_rule = color_in_rule
        self.break_win_rule = break_win_rule
        self.distractor_obj = distractor_obj
        self.distractor_rule_block = distractor_rule_block
        self.irrelevant_rule_distractor = irrelevant_rule_distractor
        # add a distractor active win rule (not just a rule block for the distractor object)
        self.distractor_win_rule = distractor_win_rule

        self.win_obj_set = win_obj_set if win_obj_set is not None else NAMES

        super().__init__(width=width, height=height, **kwargs)

    def _sample_objects(self):
            # Sample the objects
        if self.color_in_rule:
            # obj1 and obj2 can be of the same type but different colors
            obj1_idx, obj2_idx, obj3_idx = np.random.choice(len(OBJECTS), size=3, replace=False)
            obj1 = OBJECTS[obj1_idx]
            obj2 = OBJECTS[obj2_idx]
            # obj3 only useful when distractor_rule_block is True and irrelevant_rule_distractor is True
            obj3 = OBJECTS[obj3_idx]
        else:
            # make sure obj1 and obj2 are different object types
            obj1_name, obj2_name, obj3_name = np.random.choice(len(NAMES), size=3, replace=False)
            obj1_color, obj2_color, obj3_color = np.random.choice(len(COLORS), size=3, replace=True)
            obj1 = (COLORS[obj1_color], NAMES[obj1_name])
            obj2 = (COLORS[obj2_color], NAMES[obj2_name])
            obj3 = (COLORS[obj3_color], NAMES[obj3_name])
        return obj1, obj2, obj3

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Sample the objects
        obj1, obj2, obj3 = self._sample_objects()
        # TODO: constraint only object name (don't support color constraint yet)
        while obj1[1] not in self.win_obj_set:
            obj1, obj2, obj3 = self._sample_objects()

        # Add the rules
        put_rule(self, "baba", "you", positions=(1, 6))

        if self.color_in_rule:
            win_obj = obj1
            block_idx = 1
        else:
            win_obj = obj1[1]
            block_idx = 0
        # add win rule
        put_rule(self, win_obj, "win", positions=(1, 1))
        # break the win rule
        if self.break_win_rule:
            break_rule(self, (win_obj, "win"), new_pos={"top": (2, 2), "size": (4, 4)}, block_idx=block_idx)

        if self.distractor_rule_block:
            # add distractor rule block for the other object
            # but place it such that it is impossible to make the obj2 win
            positions = [
                (4, 6), (5, 6), (6, 6),  # bottom row
                (4, 1), (5, 1), (6, 1),  # top row
                (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)   # right column
            ]
            pos = positions[np.random.choice(len(positions))]

            if self.irrelevant_rule_distractor:
                # add a rule block that is different from obj1 and obj2
                put_obj(self, RuleObject(obj3[1]), pos)
            else:
                put_obj(self, RuleObject(obj2[1]), pos)

        # Place the objects and agent in the grid
        place_obj(self, "baba", top=(1, 2))
        place_obj(self, obj1, top=(1, 2))
        if self.distractor_obj:
            place_obj(self, obj2, top=(1, 2))

        # add extra distractor win rule
        if self.distractor_win_rule:
            put_rule(self, obj3[1], "win", positions=(4, 4))

        if self.color_in_rule:
            self.win_rule = f"{win_obj[0]} {win_obj[1]} is win"
        else:
            self.win_rule = f"{win_obj} is win"
        self.win_obj = win_obj

        if self.break_win_rule:
            self.target_plan = f"make[{self.win_rule}], goto[{self.win_obj}]" if self.color_in_rule else f"make[{self.win_rule}], goto[{self.win_obj}]"
        else:
            self.target_plan = f"goto[{self.win_obj}]"


@register("env/goto_win-distr_obj_rule")
class GotoWinEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, **kwargs)

@register("env/goto_win")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=False, distractor_rule_block=False, **kwargs)

@register("env/goto_win-distr_obj")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=True, distractor_rule_block=False, **kwargs)

@register("env/goto_win-distr_rule")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=False, distractor_rule_block=True, **kwargs)

@register("env/goto_win-distr_obj-irrelevant_rule")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


@register("env/goto_win-distr_win_rule")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=True, distractor_win_rule=True, **kwargs)


@register("env/make_win-distr_obj")
class MakeWinNoDistractorRuleEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(distractor_rule_block=False, **kwargs)

@register("env/make_win-distr_rule")
class MakeWinNoDistractorObjEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(distractor_obj=False, **kwargs)

@register("env/make_win")
class MakeWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(distractor_obj=False, distractor_rule_block=False, **kwargs)

@register("env/make_win-distr_obj-irrelevant_rule")
class MakeWinIrrelevantDistractorRuleEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


# ===== Single-room make win splits =====
env_ids = [
    "env/make_win",
    "env/make_win-distr_obj",
    "env/make_win-distr_rule",
    "env/make_win-distr_obj-irrelevant_rule",
    "env/make_win-distr_obj_rule",
]
for env_id in env_ids:
    # NAMES minus "ball"
    win_obj_set = [name for name in NAMES if name != "ball"]
    register(
        f"{env_id}#no_ball_win",
        partial(make, env_id, win_obj_set=win_obj_set)
    )
    register(
        f"{env_id}#only_ball_win",
        partial(make, env_id, win_obj_set=["ball"]),
    )



class TwoRoomEnv(BabaIsYouEnv):
    def __init__(
            self,
            width=13,
            height=9,
            baba_pos="left_pushable",
            obj1_pos="anywhere",
            obj2_pos="anywhere",
            break_stop_rule=False,
            break_win_rule=False,
            distractor_obj=True,
            distractor_rule_block=True,
            irrelevant_rule_distractor=False,
            color_in_rule=False,
            distractor_win_rule=False,
            **kwargs
        ):
        self.color_in_rule = color_in_rule
        self.break_stop_rule = break_stop_rule
        self.break_win_rule = break_win_rule
        self.distractor_obj = distractor_obj
        self.distractor_rule_block = distractor_rule_block
        self.irrelevant_rule_distractor = irrelevant_rule_distractor
        self.distractor_win_rule = distractor_win_rule

        pushable_area_size = (width // 2 - 2, height - 6)
        self.positions = {
            # not all left or right space because want it to be possible to push the objects
            "left_pushable": {
                "top": (2, 3),
                "size": pushable_area_size
            },
            "right_pushable": {
                "top": (width // 2 + 1, 3),
                "size": pushable_area_size
            },
            "right_unpushable": [
                # left border
                *[(width-2, 2+i) for i in range(height-4)],
                # bottom border
                *[(width-2-i, height-2) for i in range(width//2-2)],
            ],
            "left_anywhere": {
                "top": (1, 1),
                "size": (width // 2 - 2, height - 2)
            },
            "right_anywhere": {
                # start at y = 2 to prevent having object at the same place as the obj rule block of the win rule for the make_win envs
                "top": (width // 2 + 1, 2),
                "size": (width // 2 - 2, height - 2)
            },
            "anywhere": {
                "top": (1, 2),
                "size": (width - 2, height - 2)
            }
        }
        self.left_pushable = self.positions["left_pushable"]
        self.right_pushable = self.positions["right_pushable"]

        self.baba_pos = self.positions[baba_pos]
        self.obj1_pos = self.positions[obj1_pos]
        self.obj2_pos = self.positions[obj2_pos]

        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Add the vertical wall dividing the two rooms
        self.grid.vert_wall(width // 2, 1, height - 2, obj_type=lambda: make_obj("wall"))

        # Sample the objects
        if self.color_in_rule:
            # obj1 and obj2 can be of the same type but different colors
            obj1_idx, obj2_idx, obj3_idx = np.random.choice(len(OBJECTS), size=3, replace=False)
            obj1 = OBJECTS[obj1_idx]
            obj2 = OBJECTS[obj2_idx]
            # obj3 only useful when distractor_rule_block is True and irrelevant_rule_distractor is True
            obj3 = OBJECTS[obj3_idx]
        else:
            # make sure obj1 and obj2 are different object types
            obj1_name, obj2_name, obj3_name = np.random.choice(len(NAMES), size=3, replace=False)
            obj1_color, obj2_color, obj3_color = np.random.choice(len(COLORS), size=3, replace=True)
            obj1 = (COLORS[obj1_color], NAMES[obj1_name])
            obj2 = (COLORS[obj2_color], NAMES[obj2_name])
            obj3 = (COLORS[obj3_color], NAMES[obj3_name])

        # Add the rules
        # put "baba is you" such that it cannot be changed
        put_rule(self, "baba", "you", positions=(1, height - 2))
        # the agent should be able to break "wall is stop" if needed

        # with this placement, the agent can make "wall is win" and goto "wall"
        # put_rule(self, "wall", "stop", positions=(2, 2))
        # to avoid that we add an unpushable block
        put_rule(self, "wall", "stop", positions=(1, 2))
        # put_obj(self, Wall(), pos=(1, 3))

        if self.color_in_rule:
            # put "obj1 is win" in the corner so that can place a distractor rule block for obj2 such it's impossible to make obj2 win
            put_rule(self, obj1, "win", positions=(width - 4, 1))
        else:
            put_rule(self, obj1[1], "win", positions=(width - 4, 1))

        # Add the distractor rule block
        if self.distractor_rule_block:
            if self.irrelevant_rule_distractor:
                # add a rule block that is different from obj1 and obj2
                place_obj(self, RuleObject(obj3[1]), **self.positions["right_pushable"])
            elif self.distractor_obj:                
                # add distractor rule block for the other object
                # but place it such that it is impossible to make the obj2 win
                positions = self.positions["right_unpushable"]
                pos = positions[np.random.choice(len(positions))]
                put_obj(self, RuleObject(obj2[1]), pos)
            else:
                # don't need to be unpushable because obj2 is not in the env
                place_obj(self, RuleObject(obj2[1]), **self.positions["right_pushable"])

        # Break the rules if needed
        if self.break_stop_rule:
            break_rule(self, ("wall", "stop"), new_pos=self.left_pushable, block_idx=[1, 2])

        if self.break_win_rule:
            win_obj = obj1 if self.color_in_rule else obj1[1]
            block_idx = 1 if self.color_in_rule else 0
            break_rule(self, (win_obj, "win"), new_pos=self.right_pushable, block_idx=block_idx)

        # Place the objects and agent in the rooms
        baba_pos = place_obj(self, "baba", **self.baba_pos)
        obj1_pos = place_obj(self, obj1, **self.obj1_pos)
        if self.distractor_obj:
            place_obj(self, obj2, **self.obj2_pos)

        # add extra distractor win rule
        if self.distractor_win_rule:
            put_rule(self, obj3[1], "win", positions=(width-4, 4))

        # if win rule active and win object on the left, plan = goto win object
        if not self.break_win_rule and obj1_pos[0] < width // 2:
            self.target_plan = f"goto[{obj1[1]}]"
        # if win rule active, win obj on the right and stop rule inactive, plan = goto win object
        elif not self.break_win_rule and obj1_pos[0] > width // 2 and self.break_stop_rule:
            self.target_plan = f"goto[{obj1[1]}]"
        # if win rule active, win obj on the right and stop rule active, plan = break stop rule, goto win object
        elif not self.break_win_rule and obj1_pos[0] > width // 2 and not self.break_stop_rule:
            self.target_plan = f"break[wall is stop], goto[{obj1[1]}]"
        # if win rule inactive and stop rule inactive, plan = make win rule, goto win obj
        elif self.break_win_rule and self.break_stop_rule:
            self.target_plan = f"make[{obj1[1]} is win], goto[{obj1[1]}]"
        # if win rule inactive and stop rule active, plan = break stop rule, make win rule, goto win obj
        elif self.break_win_rule and not self.break_stop_rule:
            self.target_plan = f"break[wall is stop], make[{obj1[1]} is win], goto[{obj1[1]}]"
        else:
            breakpoint()
            raise ValueError("Invalid configuration")


@register("env/two_room-goto_win")
class TwoRoomGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, distractor_rule_block=False, break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_obj_rule")
class TwoRoomGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_rule")
class TwoRoomGotoWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_obj")
class TwoRoomGotoWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_rule_block=False, break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_obj-irrelevant_rule")
class TwoRoomGotoWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True, break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_win_rule")
class TwoRoomGotoWinDistrWinRule(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_win_rule=True, break_stop_rule=True, **kwargs)


# ===== variants of break stop, goto win =====
@register("env/two_room-break_stop-goto_win-distr_obj_rule")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", **kwargs)


@register("env/two_room-break_stop-goto_win-distr_obj")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_rule_block=False, **kwargs)


@register("env/two_room-break_stop-goto_win-distr_rule")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, **kwargs)

@register("env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


@register("env/two_room-break_stop-goto_win")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, distractor_rule_block=False, **kwargs)


@register("env/two_room-maybe_break_stop-goto_win-distr_obj_rule")
class TwoRoomMaybeBreakStopGotoWinDistrObjRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", **kwargs)


@register("env/two_room-maybe_break_stop-goto_win")
class TwoRoomMaybeBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=False, distractor_rule_block=False, **kwargs)


@register("env/two_room-maybe_break_stop-goto_win-distr_obj")
class TwoRoomMaybeBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=True, distractor_rule_block=False, **kwargs)


@register("env/two_room-maybe_break_stop-goto_win-distr_rule")
class TwoRoomMaybeBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=False, distractor_rule_block=True, **kwargs)


@register("env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule")
class TwoRoomMaybeBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


# ===== variants of make win =====
@register("env/two_room-make_win-distr_obj_rule")
class TwoRoomMakeWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", **kwargs)


@register("env/two_room-make_win-distr_rule")
class TwoRoomMakeWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, **kwargs)


@register("env/two_room-make_win")
class TwoRoomMakeWinNoDistractorEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, distractor_rule_block=False, **kwargs)


@register("env/two_room-make_win-distr_obj-irrelevant_rule")
class TwoRoomMakeWinIrrelevantDistractorRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


@register("env/two_room-make_win-distr_obj")
class TwoRoomMakeWinNoDistractorRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_rule_block=False, **kwargs)


@register("env/two_room-make_win-distr_win_rule")
class TwoRoomGotoWinDistrWinRule(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_win_rule=True, break_win_rule=True, break_stop_rule=True, **kwargs)


# ===== variants of break stop, make win =====
@register("env/two_room-break_stop-make_win-distr_obj_rule")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", **kwargs)


@register("env/two_room-break_stop-make_win-distr_rule")
class TwoRoomBreakStopMakeWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, **kwargs)


@register("env/two_room-break_stop-make_win")
class TwoRoomBreakStopMakeWinNoDistractorEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, distractor_rule_block=False, **kwargs)


@register("env/two_room-break_stop-make_win-distr_obj-irrelevant_rule")
class TwoRoomBreakStopMakeWinIrrelevantDistractorRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


@register("env/two_room-break_stop-make_win-distr_obj")
class TwoRoomBreakStopMakeWinNoDistractorRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_rule_block=False, **kwargs)


@register("env/two_room-make_you")
class TwoRoomMakeYouEnv(BabaIsYouEnv):
    def __init__(
            self,
            width=13,
            height=9,
            baba_pos="left_pushable",
            obj1_pos="right_anywhere",
            obj2_pos="right_anywhere",
            break_stop_rule=False,
            break_win_rule=False,
            distractor_obj=True,
            distractor_rule_block=True,
            irrelevant_rule_distractor=False,
            color_in_rule=False,
            **kwargs
        ):
        self.color_in_rule = color_in_rule
        self.break_stop_rule = break_stop_rule
        self.break_win_rule = break_win_rule
        self.distractor_obj = distractor_obj
        self.distractor_rule_block = distractor_rule_block
        self.irrelevant_rule_distractor = irrelevant_rule_distractor

        pushable_area_size = (width // 2 - 3, height - 6)
        self.positions = {
            # not all left or right space because want it to be possible to push the objects
            "left_pushable": {
                "top": (2, 3),
                "size": pushable_area_size
            },
            "right_pushable": {
                "top": (width // 2 + 2, 3),
                "size": pushable_area_size
            },
            "right_unpushable": [
                # left border
                *[(width-2, 2+i) for i in range(height-4)],
                # bottom border
                *[(width-2-i, height-2) for i in range(width//2-2)],
            ],
            "left_anywhere": {
                "top": (1, 1),
                "size": (width // 2 - 2, height - 2)
            },
            "right_anywhere": {
                # start at y = 2 to prevent having object at the same place as the obj rule block of the win rule for the make_win envs
                "top": (width // 2 + 1, 2),
                "size": (width // 2 - 2, height - 2)
            },
            "anywhere": {
                "top": (1, 2),
                "size": (width - 2, height - 2)
            }
        }
        self.left_pushable = self.positions["left_pushable"]
        self.right_pushable = self.positions["right_pushable"]

        self.baba_pos = self.positions[baba_pos]
        self.obj1_pos = self.positions[obj1_pos]
        self.obj2_pos = self.positions[obj2_pos]

        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Add the vertical wall dividing the two rooms
        self.grid.vert_wall(width // 2, 1, height - 2, obj_type=lambda: make_obj("wall"))

        # Sample the objects
        if self.color_in_rule:
            # obj1 and obj2 can be of the same type but different colors
            obj1_idx, obj2_idx, obj3_idx = np.random.choice(len(OBJECTS), size=3, replace=False)
            obj1 = OBJECTS[obj1_idx]
            obj2 = OBJECTS[obj2_idx]
            # obj3 only useful when distractor_rule_block is True and irrelevant_rule_distractor is True
            obj3 = OBJECTS[obj3_idx]
        else:
            # make sure obj1 and obj2 are different object types
            obj1_name, obj2_name, obj3_name = np.random.choice(len(NAMES), size=3, replace=False)
            obj1_color, obj2_color, obj3_color = np.random.choice(len(COLORS), size=3, replace=True)
            obj1 = (COLORS[obj1_color], NAMES[obj1_name])
            obj2 = (COLORS[obj2_color], NAMES[obj2_name])
            obj3 = (COLORS[obj3_color], NAMES[obj3_name])

        # Add the rules
        # put "baba is you" such that it cannot be changed
        # put_rule(self, "baba", "you", positions=(1, height - 2))
        put_rule(self, "baba", "you", positions=(1, height - 3))
        # the agent should be able to break "wall is stop" if needed

        # with this placement, the agent can make "wall is win" and goto "wall"
        # put_rule(self, "wall", "stop", positions=(2, 2))
        # to avoid that we add an unpushable block
        put_rule(self, "wall", "stop", positions=(1, 1))

        if self.color_in_rule:
            # put "obj1 is win" in the corner so that can place a distractor rule block for obj2 such it's impossible to make obj2 win
            put_rule(self, obj1, "win", positions=(width - 4, 1))
        else:
            put_rule(self, obj1[1], "win", positions=(width - 4, 1))

        # Add the distractor rule block
        if self.distractor_rule_block:
            if self.irrelevant_rule_distractor:
                # add a rule block that is different from obj1 and obj2
                place_obj(self, RuleObject(obj3[1]), **self.positions["left_pushable"])
            elif self.distractor_obj:                
                place_obj(self, RuleObject(obj2[1]), **self.positions["left_pushable"])
            else:
                # don't need to be unpushable because obj2 is not in the env
                place_obj(self, RuleObject(obj2[1]), **self.positions["left_pushable"])

        # Break the rules if needed
        if self.break_stop_rule:
            break_rule(self, ("wall", "stop"), new_pos=self.left_pushable, block_idx=[1, 2])

        if self.break_win_rule:
            win_obj = obj1 if self.color_in_rule else obj1[1]
            block_idx = 1 if self.color_in_rule else 0
            break_rule(self, (win_obj, "win"), new_pos=self.right_pushable, block_idx=block_idx)

        # Place the objects and agent in the rooms
        place_obj(self, "baba", **self.baba_pos)
        place_obj(self, obj1, **self.obj1_pos)
        if self.distractor_obj:
            place_obj(self, obj2, **self.obj2_pos)

        if self.break_win_rule:
            self.target_plan = f"make[{obj2[1]} is you], make[{obj1[1]} is win], goto[{obj1[1]}]"
        else:
            self.target_plan = f"make[{obj2[1]} is you], goto[{obj1[1]}]"


@register("env/two_room-make_you-make_win")
class TwoRoomMakeWinMakeWinEnv(TwoRoomMakeYouEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=True, **kwargs)


@register("env/two_room-make_wall_win")
class TwoRoomMakeWallWinEnv(BabaIsYouEnv):
    def __init__(
            self,
            width=13,
            height=9,
            baba_pos="left_pushable",
            obj1_pos="right_anywhere",
            obj2_pos="right_anywhere",
            break_stop_rule=False,
            distractor_obj=True,
            distractor_rule_block=True,
            irrelevant_rule_distractor=True,
            **kwargs
        ):
        self.break_stop_rule = break_stop_rule
        self.distractor_obj = distractor_obj
        self.distractor_rule_block = distractor_rule_block
        self.irrelevant_rule_distractor = irrelevant_rule_distractor

        pushable_area_size = (width // 2 - 3, height - 6)
        self.positions = {
            # not all left or right space because want it to be possible to push the objects
            "left_pushable": {
                "top": (2, 3),
                "size": pushable_area_size
            },
            "right_pushable": {
                "top": (width // 2 + 2, 3),
                "size": pushable_area_size
            },
            "right_unpushable": [
                # left border
                *[(width-2, 2+i) for i in range(height-4)],
                # bottom border
                *[(width-2-i, height-2) for i in range(width//2-2)],
            ],
            "left_anywhere": {
                "top": (1, 1),
                "size": (width // 2 - 2, height - 2)
            },
            "right_anywhere": {
                # start at y = 2 to prevent having object at the same place as the obj rule block of the win rule for the make_win envs
                "top": (width // 2 + 1, 2),
                "size": (width // 2 - 2, height - 2)
            },
            "anywhere": {
                "top": (1, 2),
                "size": (width - 2, height - 2)
            }
        }
        self.left_pushable = self.positions["left_pushable"]
        self.right_pushable = self.positions["right_pushable"]

        self.baba_pos = self.positions[baba_pos]
        self.obj1_pos = self.positions[obj1_pos]
        self.obj2_pos = self.positions[obj2_pos]

        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Add the vertical wall dividing the two rooms
        self.grid.vert_wall(width // 2, 1, height - 2, obj_type=lambda: make_obj("wall"))

        # Sample the objects
        obj1_name, obj2_name, obj3_name = np.random.choice(len(NAMES), size=3, replace=False)
        obj1_color, obj2_color, obj3_color = np.random.choice(len(COLORS), size=3, replace=True)
        obj1 = (COLORS[obj1_color], NAMES[obj1_name])
        obj2 = (COLORS[obj2_color], NAMES[obj2_name])
        obj3 = (COLORS[obj3_color], NAMES[obj3_name])

        put_rule(self, "baba", "you", positions=(1, height - 2))
        put_rule(self, "wall", "stop", positions=(2, 2))
        put_rule(self, obj1[1], "win", positions=(width - 4, 1))

        if self.distractor_rule_block:
            if self.irrelevant_rule_distractor:
                # add a rule block that is different from obj1 and obj2
                place_obj(self, RuleObject(obj3[1]), **self.positions["right_pushable"])
            elif self.distractor_obj:
                # add distractor rule block for the other object
                # but place it such that it is impossible to make the obj2 win
                positions = self.positions["right_unpushable"]
                pos = positions[np.random.choice(len(positions))]
                put_obj(self, RuleObject(obj2[1]), pos)
            else:
                # don't need to be unpushable because obj2 is not in the env
                place_obj(self, RuleObject(obj2[1]), **self.positions["right_pushable"])

        # Break the rules if needed
        if self.break_stop_rule:
            break_rule(self, ("wall", "stop"), new_pos=self.left_pushable, block_idx=[1, 2])

        new_pos = self.positions["right_unpushable"][np.random.choice(len(self.positions["right_unpushable"]))]
        break_rule(self, (obj1[1], "win"), new_pos=new_pos, block_idx=0)

        # Place the objects and agent in the rooms
        place_obj(self, "baba", **self.baba_pos)
        place_obj(self, obj1, **self.obj1_pos)
        if self.distractor_obj:
            place_obj(self, obj2, **self.obj2_pos)

        self.target_plan = f"break[wall is stop], make[wall is win], goto[wall]"


class SeededRuleCompositionEnv(BabaIsYouEnv):
    """Base class for reproducible families of rule-composition levels."""

    def __init__(self, width=8, height=8, **kwargs):
        self._composition_rng = np.random.RandomState()
        self.generation_seed = None
        super().__init__(width=width, height=height, **kwargs)

    def seed(self, seed=None):
        """Seed this generator, including through BALROG's legacy Gym wrapper."""
        self.generation_seed = None if seed is None else int(seed)
        self._composition_rng = np.random.RandomState(seed)
        return [seed]

    def reset(self, *, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed)
        # This family uses its own RNG so it also works through Gym v0.21's
        # seed-then-reset compatibility path. Avoid storing Gym's Generator,
        # which cannot be deep-copied with Gym 0.25 and NumPy 2.x.
        return super().reset(seed=None, return_info=return_info, options=options)

    def _choice(self, values):
        return values[int(self._composition_rng.randint(len(values)))]

    def _sample_objects(self, count=2):
        names = [str(name) for name in self._composition_rng.permutation(NAMES)[:count]]
        colors = [str(color) for color in self._composition_rng.choice(COLORS, size=count)]
        return list(zip(colors, names))

    def _init_generated_grid(self):
        self.grid = BabaIsYouGrid(self.width, self.height)
        self.grid.wall_rect(0, 0, self.width, self.height)

        # Keeping this rule against the left and bottom borders makes it
        # impossible to displace without giving text blocks special behavior.
        put_rule(self, "baba", "you", positions=(1, self.height - 2))

    def _set_solution(self, target_plan, actions, **generation_params):
        self.target_plan = target_plan
        # Used by tests and offline validation; BALROG does not expose it.
        self.reference_solution = tuple(actions)
        self.generation_params = {
            "seed": self.generation_seed,
            **generation_params,
        }

    def _transform_square_pos(self, pos, *, flip_vertical=False, transpose=False):
        """Transform a canonical layout while preserving its rule-token order.

        The transformed generators define their canonical rules horizontally;
        vertical reflection and transposition therefore keep the words ordered.
        """
        x, y = pos
        if flip_vertical:
            y = self.height - 1 - y
        if transpose:
            x, y = y, x
        return x, y

    def _transform_square_actions(self, actions, *, flip_vertical=False, transpose=False):
        """Apply the same geometric transform to a reference action sequence."""
        vectors = {name: np.array(delta) for name, delta in ACTIONS.items()}
        names = {tuple(delta): name for name, delta in ACTIONS.items()}
        transformed = []
        for action in actions:
            dx, dy = vectors[action]
            if flip_vertical:
                dy = -dy
            if transpose:
                dx, dy = dy, dx
            transformed.append(names[(int(dx), int(dy))])
        return transformed


@register("env/composition-01-target-lock")
class TargetLockEnv(SeededRuleCompositionEnv):
    """A target is both WIN and STOP; remove only the STOP rule."""

    def _gen_grid(self, width, height, params=None):
        self._init_generated_grid()
        target, distractor = self._sample_objects()
        target_color, target_name = target
        distractor_color, distractor_name = distractor

        variants = [
            {
                "rule": (2, 2),
                "baba": (4, 3),
                "target": (6, 3),
                "distractor": (1, 5),
                "actions": ["up", "right", "right", "down"],
            },
            {
                "rule": (3, 2),
                "baba": (5, 3),
                "target": (1, 3),
                "distractor": (6, 5),
                "actions": ["up", "down", "left", "left", "left", "left"],
            },
            {
                "rule": [(2, 1), (2, 2), (2, 3)],
                "baba": (3, 3),
                "target": (6, 4),
                "distractor": (1, 5),
                "actions": ["left", "right", "right", "right", "right", "down"],
            },
            {
                "rule": [(5, 1), (5, 2), (5, 3)],
                "baba": (4, 3),
                "target": (1, 4),
                "distractor": (6, 5),
                "actions": ["right", "left", "left", "left", "left", "down"],
            },
        ]
        variant_idx = int(self._composition_rng.randint(len(variants)))
        variant = variants[variant_idx]

        put_rule(self, target_name, "stop", positions=variant["rule"])
        put_rule(self, target_name, "win", positions=(4, height - 2))
        put_obj(self, "baba", variant["baba"])
        put_obj(self, target, variant["target"])
        put_obj(self, distractor, variant["distractor"])

        self._set_solution(
            f"break[{target_name} is stop], preserve[{target_name} is win], goto[{target_name}]",
            variant["actions"],
            variant=variant_idx,
            target=target_name,
            target_color=target_color,
            distractor=distractor_name,
            distractor_color=distractor_color,
        )


@register("env/composition-02-stop-to-win")
class StopToWinEnv(SeededRuleCompositionEnv):
    """Replace the target's STOP property with WIN."""

    def _gen_grid(self, width, height, params=None):
        self._init_generated_grid()
        target, distractor = self._sample_objects()
        target_color, target_name = target
        distractor_color, distractor_name = distractor

        variants = [
            {
                "rule": (2, 3),
                "win": (4, 2),
                "baba": (4, 1),
                "target": (6, 2),
                "distractor": (1, 5),
                "actions": ["down", "right", "right"],
            },
            {
                "rule": (2, 3),
                "win": (4, 4),
                "baba": (4, 5),
                "target": (1, 4),
                "distractor": (6, 5),
                "actions": ["up", "left", "left", "left"],
            },
            {
                "rule": [(3, 2), (3, 3), (3, 4)],
                "win": (2, 4),
                "baba": (1, 4),
                "target": (2, 1),
                "distractor": (6, 5),
                "actions": ["right", "up", "up", "up"],
            },
            {
                "rule": [(4, 1), (4, 2), (4, 3)],
                "win": (5, 3),
                "baba": (6, 3),
                "target": (6, 5),
                "distractor": (1, 5),
                "actions": ["left", "right", "down", "down"],
            },
        ]
        variant_idx = int(self._composition_rng.randint(len(variants)))
        variant = variants[variant_idx]

        put_rule(self, target_name, "stop", positions=variant["rule"])
        put_obj(self, RuleProperty("win"), variant["win"])
        put_obj(self, "baba", variant["baba"])
        put_obj(self, target, variant["target"])
        put_obj(self, distractor, variant["distractor"])

        self._set_solution(
            f"replace[{target_name} is stop -> {target_name} is win], goto[{target_name}]",
            variant["actions"],
            variant=variant_idx,
            target=target_name,
            target_color=target_color,
            distractor=distractor_name,
            distractor_color=distractor_color,
        )


@register("env/composition-03-reuse-is")
class ReuseIsEnv(SeededRuleCompositionEnv):
    """Reuse IS from WALL IS STOP to complete a target WIN rule."""

    def __init__(self, width=13, height=9, **kwargs):
        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        self._init_generated_grid()
        target, distractor = self._sample_objects()
        target_color, target_name = target
        distractor_color, distractor_name = distractor
        target_x = int(self._choice([9, 10, 11]))

        variants = [
            {
                "source_rule": (1, 2),
                "target_subject": (1, 5),
                "target_win": (3, 5),
                "baba": (2, 1),
                "target": (target_x, 4),
                "actions": ["down"] * 3 + ["right"] * (target_x - 2),
            },
            {
                "source_rule": (1, 5),
                "target_subject": (1, 2),
                "target_win": (3, 2),
                "baba": (2, 6),
                "target": (target_x, 3),
                "actions": ["up"] * 3 + ["right"] * (target_x - 2),
            },
            {
                "source_rule": [(2, 1), (2, 2), (2, 3)],
                "target_subject": (4, 1),
                "target_win": (4, 3),
                "baba": (1, 2),
                "target": (target_x, 4),
                "actions": ["right"] * 2 + ["down"] * 2 + ["right"] * (target_x - 3),
            },
            {
                "source_rule": [(4, 1), (4, 2), (4, 3)],
                "target_subject": (2, 1),
                "target_win": (2, 3),
                "baba": (5, 2),
                "target": (target_x, 2),
                "actions": ["left"] * 2 + ["right"] * (target_x - 3),
            },
        ]
        variant_idx = int(self._composition_rng.randint(len(variants)))
        variant = variants[variant_idx]

        put_rule(self, "wall", "stop", positions=variant["source_rule"])
        put_obj(self, RuleObject(target_name), variant["target_subject"])
        put_obj(self, RuleProperty("win"), variant["target_win"])
        self.grid.vert_wall(width // 2, 1, height - 2, obj_type=lambda: make_obj("wall"))
        put_obj(self, "baba", variant["baba"])
        put_obj(self, target, variant["target"])
        put_obj(self, distractor, self._choice([(8, 1), (8, 6), (11, 6)]))

        self._set_solution(
            f"break[wall is stop], reuse[is], make[{target_name} is win], goto[{target_name}]",
            variant["actions"],
            variant=variant_idx,
            target=target_name,
            target_color=target_color,
            target_x=target_x,
            distractor=distractor_name,
            distractor_color=distractor_color,
        )


@register("env/composition-04-reuse-win")
class ReuseWinEnv(SeededRuleCompositionEnv):
    """Move WIN from an unusable object rule to the target rule."""

    def _gen_grid(self, width, height, params=None):
        self._init_generated_grid()
        target, wrong_target, distractor = self._sample_objects(count=3)
        target_color, target_name = target
        _, wrong_target_name = wrong_target
        distractor_color, distractor_name = distractor

        variants = [
            {
                "source_rule": (1, 2),
                "target_subject": (1, 4),
                "target_is": (2, 4),
                "baba": (3, 1),
                "target": (6, 3),
                "distractor": (1, 5),
                "actions": ["down", "down", "right", "right", "right"],
            },
            {
                "source_rule": (1, 4),
                "target_subject": (1, 2),
                "target_is": (2, 2),
                "baba": (3, 5),
                "target": (6, 3),
                "distractor": (6, 5),
                "actions": ["up", "up", "right", "right", "right"],
            },
            {
                "source_rule": [(2, 1), (2, 2), (2, 3)],
                "target_subject": (4, 1),
                "target_is": (4, 2),
                "baba": (1, 3),
                "target": (6, 4),
                "distractor": (1, 5),
                "actions": ["right", "right", "down", "right", "right", "right"],
            },
            {
                "source_rule": [(5, 1), (5, 2), (5, 3)],
                "target_subject": (3, 1),
                "target_is": (3, 2),
                "baba": (6, 3),
                "target": (1, 4),
                "distractor": (6, 5),
                "actions": ["left", "left", "down", "left", "left", "left"],
            },
        ]
        variant_idx = int(self._composition_rng.randint(len(variants)))
        variant = variants[variant_idx]

        put_rule(self, wrong_target_name, "win", positions=variant["source_rule"])
        put_obj(self, RuleObject(target_name), variant["target_subject"])
        put_obj(self, RuleIs(), variant["target_is"])
        put_obj(self, "baba", variant["baba"])
        put_obj(self, target, variant["target"])
        put_obj(self, distractor, variant["distractor"])

        self._set_solution(
            f"break[{wrong_target_name} is win], reuse[win], make[{target_name} is win], goto[{target_name}]",
            variant["actions"],
            variant=variant_idx,
            target=target_name,
            target_color=target_color,
            wrong_target=wrong_target_name,
            distractor=distractor_name,
            distractor_color=distractor_color,
        )


@register("env/composition-05-cross-rule")
class CrossRuleEnv(SeededRuleCompositionEnv):
    """Break WALL IS STOP while preserving the crossing target WIN rule."""

    def __init__(self, width=13, height=9, **kwargs):
        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        self._init_generated_grid()
        target, distractor = self._sample_objects()
        target_color, target_name = target
        distractor_color, distractor_name = distractor
        target_x = int(self._choice([9, 10, 11]))

        variants = [
            {
                "wall": (2, 3),
                "is": (3, 3),
                "stop": (4, 3),
                "target_word": (3, 2),
                "win": (3, 4),
                "baba": (4, 4),
                "target": (target_x, 3),
                "actions": ["up"] + ["right"] * (target_x - 4),
            },
            {
                "wall": (2, 5),
                "is": (3, 5),
                "stop": (4, 5),
                "target_word": (3, 4),
                "win": (3, 6),
                "baba": (4, 4),
                "target": (target_x, 5),
                "actions": ["down"] + ["right"] * (target_x - 4),
            },
            {
                "wall": (3, 2),
                "is": (3, 3),
                "stop": (3, 4),
                "target_word": (2, 3),
                "win": (4, 3),
                "baba": (2, 4),
                "target": (target_x, 5),
                "actions": ["right", "down"] + ["right"] * (target_x - 3),
            },
            {
                "wall": (4, 2),
                "is": (4, 3),
                "stop": (4, 4),
                "target_word": (3, 3),
                "win": (5, 3),
                "baba": (5, 4),
                "target": (target_x, 4),
                "actions": ["left"] + ["right"] * (target_x - 4),
            },
        ]
        variant_idx = int(self._composition_rng.randint(len(variants)))
        variant = variants[variant_idx]

        put_obj(self, RuleObject("wall"), variant["wall"])
        put_obj(self, RuleIs(), variant["is"])
        put_obj(self, RuleProperty("stop"), variant["stop"])
        put_obj(self, RuleObject(target_name), variant["target_word"])
        put_obj(self, RuleProperty("win"), variant["win"])
        self.grid.vert_wall(width // 2, 1, height - 2, obj_type=lambda: make_obj("wall"))
        put_obj(self, "baba", variant["baba"])
        put_obj(self, target, variant["target"])
        put_obj(self, distractor, self._choice([(8, 1), (8, 6), (11, 6)]))

        self._set_solution(
            f"break[wall is stop], preserve[{target_name} is win], goto[{target_name}]",
            variant["actions"],
            variant=variant_idx,
            target=target_name,
            target_color=target_color,
            target_x=target_x,
            distractor=distractor_name,
            distractor_color=distractor_color,
        )


@register("env/composition-06-color-scope")
class ColorScopeEnv(SeededRuleCompositionEnv):
    """Narrow OBJECT IS STOP with a color prefix to pass an off-color object."""

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        blocker, goal = self._sample_objects(count=2)
        blocker_color, blocker_name = blocker
        goal_color, goal_name = goal
        scope_color = self._choice([color for color in COLORS if color != blocker_color])
        scope_twin = (scope_color, blocker_name)

        variant_idx = int(self._composition_rng.randint(4))
        flip_vertical = bool(variant_idx % 2)
        transpose = bool(variant_idx // 2)

        def transform(pos):
            return self._transform_square_pos(
                pos,
                flip_vertical=flip_vertical,
                transpose=transpose,
            )

        scope_rule = [transform(pos) for pos in [(2, 1), (3, 1), (4, 1)]]
        baba_rule = [transform(pos) for pos in [(1, 6), (2, 6), (3, 6)]]
        goal_rule = [transform(pos) for pos in [(4, 6), (5, 6), (6, 6)]]
        barrier = [transform(pos) for pos in [(4, 2), (4, 3), (4, 5)]]

        put_rule(self, blocker_name, "stop", positions=scope_rule)
        put_obj(self, RuleColor(scope_color), transform((1, 2)))
        put_rule(self, "baba", "you", positions=baba_rule)
        put_rule(self, goal_name, "win", positions=goal_rule)
        for pos in barrier:
            put_obj(self, Wall(), pos)

        put_obj(self, "baba", transform((1, 3)))
        put_obj(self, blocker, transform((4, 4)))
        put_obj(self, scope_twin, transform((2, 5)))
        put_obj(self, goal, transform((6, 4)))

        actions = self._transform_square_actions(
            ["up", "down", "down", "right", "right", "right", "right", "right"],
            flip_vertical=flip_vertical,
            transpose=transpose,
        )
        self._set_solution(
            f"narrow[{blocker_name} is stop -> {scope_color} {blocker_name} is stop], "
            f"pass[{blocker_color} {blocker_name}], goto[{goal_name}]",
            actions,
            variant=variant_idx,
            blocker=blocker_name,
            blocker_color=blocker_color,
            scope_color=scope_color,
            goal=goal_name,
            goal_color=goal_color,
        )


@register("env/composition-07-one-token-fork")
class OneTokenForkEnv(SeededRuleCompositionEnv):
    """Commit one WIN block to the useful one of two incomplete rules."""

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        first, second = self._sample_objects(count=2)
        first_color, first_name = first
        second_color, second_name = second
        variant_idx = int(self._composition_rng.randint(4))
        correct_branch = variant_idx % 2
        transpose = bool(variant_idx // 2)

        target = first if correct_branch == 0 else second
        decoy = second if correct_branch == 0 else first
        target_color, target_name = target
        decoy_color, decoy_name = decoy

        def transform(pos):
            return self._transform_square_pos(pos, transpose=transpose)

        put_rule(
            self,
            "baba",
            "you",
            positions=[transform(pos) for pos in [(1, 6), (2, 6), (3, 6)]],
        )

        # Branch 0 receives WIN by pushing up; branch 1 by pushing right.
        put_obj(self, RuleObject(first_name), transform((1, 2)))
        put_obj(self, RuleIs(), transform((2, 2)))
        put_obj(self, RuleObject(second_name), transform((4, 1)))
        put_obj(self, RuleIs(), transform((4, 2)))
        put_obj(self, RuleProperty("win"), transform((3, 3)))
        for pos in [(3, 1), (4, 4), (5, 3)]:
            put_obj(self, Wall(), transform(pos))

        put_obj(self, "baba", transform((2, 4)))
        if correct_branch == 0:
            target_pos = (1, 5)
            decoy_pos = (6, 5)
            enclosure = [(5, 5), (6, 4), (5, 6)]
            actions = ["right", "up", "left", "left", "down", "down"]
        else:
            target_pos = (6, 5)
            decoy_pos = (1, 5)
            enclosure = [(2, 5), (1, 4)]
            actions = ["up", "right", "down", "down", "right", "right", "right"]

        put_obj(self, target, transform(target_pos))
        put_obj(self, decoy, transform(decoy_pos))
        for pos in enclosure:
            put_obj(self, Wall(), transform(pos))

        actions = self._transform_square_actions(actions, transpose=transpose)
        self._set_solution(
            f"choose[win -> {target_name} is win], reject[{decoy_name} is win], goto[{target_name}]",
            actions,
            variant=variant_idx,
            target=target_name,
            target_color=target_color,
            decoy=decoy_name,
            decoy_color=decoy_color,
            correct_branch=correct_branch,
        )


@register("env/composition-08-control-handoff")
class ControlHandoffEnv(SeededRuleCompositionEnv):
    """Transfer YOU to one color of another object, then control it to WIN."""

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        target, goal = self._sample_objects(count=2)
        target_color, target_name = target
        goal_color, goal_name = goal
        twin_color = self._choice([color for color in COLORS if color != target_color])
        target_twin = (twin_color, target_name)

        variants = [
            {
                "source_rule": (2, 2),
                "target_color": (1, 3),
                "target_subject": (2, 3),
                "target_is": (3, 3),
                "baba": (4, 1),
                "barrier": [(x, 4) for x in range(1, width - 1)],
                "target": (1, 5),
                "goal": (6, 5),
                "goal_rule": (4, 6),
                "target_twin": (1, 6),
                "actions": ["down"] + ["right"] * 5,
            },
            {
                "source_rule": (3, 2),
                "target_color": (2, 3),
                "target_subject": (3, 3),
                "target_is": (4, 3),
                "baba": (5, 1),
                "barrier": [(x, 4) for x in range(1, width - 1)],
                "target": (6, 5),
                "goal": (1, 5),
                "goal_rule": (1, 6),
                "target_twin": (6, 6),
                "actions": ["down"] + ["left"] * 5,
            },
            {
                "source_rule": [(2, 2), (2, 3), (2, 4)],
                "target_color": (3, 1),
                "target_subject": (3, 2),
                "target_is": (3, 3),
                "baba": (1, 4),
                "barrier": [(4, y) for y in range(1, height - 1)],
                "target": (5, 1),
                "goal": (5, 6),
                "goal_rule": [(6, 4), (6, 5), (6, 6)],
                "target_twin": (6, 1),
                "actions": ["right"] + ["down"] * 5,
            },
            {
                "source_rule": [(5, 2), (5, 3), (5, 4)],
                "target_color": (4, 1),
                "target_subject": (4, 2),
                "target_is": (4, 3),
                "baba": (6, 4),
                "barrier": [(3, y) for y in range(1, height - 1)],
                "target": (2, 6),
                "goal": (2, 1),
                "goal_rule": [(1, 4), (1, 5), (1, 6)],
                "target_twin": (1, 1),
                "actions": ["left"] + ["up"] * 5,
            },
        ]
        variant_idx = int(self._composition_rng.randint(len(variants)))
        variant = variants[variant_idx]

        put_rule(self, "baba", "you", positions=variant["source_rule"])
        put_obj(self, RuleColor(target_color), variant["target_color"])
        put_obj(self, RuleObject(target_name), variant["target_subject"])
        put_obj(self, RuleIs(), variant["target_is"])
        put_rule(self, goal_name, "win", positions=variant["goal_rule"])
        for pos in variant["barrier"]:
            put_obj(self, Wall(), pos)

        put_obj(self, "baba", variant["baba"])
        put_obj(self, target, variant["target"])
        put_obj(self, goal, variant["goal"])
        put_obj(self, target_twin, variant["target_twin"])

        self._set_solution(
            f"transfer[you: baba -> {target_color} {target_name}], "
            f"control[{target_color} {target_name}], goto[{goal_name}]",
            variant["actions"],
            variant=variant_idx,
            target=target_name,
            target_color=target_color,
            goal=goal_name,
            goal_color=goal_color,
            target_twin_color=twin_color,
        )


@register("env/composition-09-ordered-assembly")
class OrderedAssemblyEnv(SeededRuleCompositionEnv):
    """Position WIN before moving IS into a passage that it permanently seals."""

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        target, distractor = self._sample_objects(count=2)
        target_color, target_name = target
        distractor_color, distractor_name = distractor
        variant_idx = int(self._composition_rng.randint(4))
        flip_vertical = bool(variant_idx % 2)
        transpose = bool(variant_idx // 2)

        def transform(pos):
            return self._transform_square_pos(
                pos,
                flip_vertical=flip_vertical,
                transpose=transpose,
            )

        put_rule(
            self,
            "baba",
            "you",
            positions=[transform(pos) for pos in [(1, 6), (2, 6), (3, 6)]],
        )
        put_rule(
            self,
            distractor_name,
            "stop",
            positions=[transform(pos) for pos in [(2, 2), (3, 2), (4, 2)]],
        )
        put_obj(self, RuleObject(target_name), transform((2, 4)))
        put_obj(self, RuleProperty("win"), transform((4, 5)))
        for pos in [(1, 4), (5, 4), (6, 4), (2, 5)]:
            put_obj(self, Wall(), transform(pos))

        put_obj(self, "baba", transform((4, 6)))
        put_obj(self, target, transform((5, 3)))
        put_obj(self, distractor, transform((1, 5)))

        actions = self._transform_square_actions(
            [
                "up",
                "left",
                "up",
                "up",
                "left",
                "left",
                "up",
                "up",
                "right",
                "right",
                "down",
                "down",
                "right",
                "right",
            ],
            flip_vertical=flip_vertical,
            transpose=transpose,
        )
        self._set_solution(
            f"prepare[win], move[is: {distractor_name} is stop -> {target_name} is win], "
            f"goto[{target_name}]",
            actions,
            variant=variant_idx,
            target=target_name,
            target_color=target_color,
            distractor=distractor_name,
            distractor_color=distractor_color,
        )


@register("env/composition-10-control-relay")
class ControlRelayEnv(SeededRuleCompositionEnv):
    """Relay control to a colored object, restore Baba remotely, then finish as Baba."""

    def __init__(self, width=13, height=9, **kwargs):
        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        target, goal = self._sample_objects(count=2)
        target_color, target_name = target
        goal_color, goal_name = goal
        twin_color = self._choice([color for color in COLORS if color != target_color])
        target_twin = (twin_color, target_name)

        variant_idx = int(self._composition_rng.randint(4))
        source_on_left = variant_idx < 2
        flip_vertical = bool(variant_idx % 2)

        if source_on_left:
            source_rule = [(2, 2), (3, 2), (4, 2)]
            target_stem = [(1, 3), (2, 3), (3, 3)]
            source_you = (4, 3)
            baba = (4, 1)
            pocket_walls = [(3, 1), (5, 1), (5, 2)]
            goal_rule = [(1, 7), (2, 7), (3, 7)]
            goal_pos = (1, 6)
            remote_baba = (7, 7)
            remote_is = (8, 7)
            remote_you_dest = (9, 7)
            remote_you = (10, 7)
            target_pos = (11, 1)
            twin_pos = (10, 1)
        else:
            source_rule = [(8, 2), (9, 2), (10, 2)]
            target_stem = [(7, 3), (8, 3), (9, 3)]
            source_you = (10, 3)
            baba = (10, 1)
            pocket_walls = [(9, 1), (11, 1), (11, 2)]
            goal_rule = [(9, 7), (10, 7), (11, 7)]
            goal_pos = (7, 6)
            remote_baba = (1, 7)
            remote_is = (2, 7)
            remote_you_dest = (3, 7)
            remote_you = (4, 7)
            target_pos = (5, 1)
            twin_pos = (4, 1)

        def transform(pos):
            x, y = pos
            if flip_vertical:
                y = height - 1 - y
            return x, y

        put_rule(self, "baba", "you", positions=[transform(pos) for pos in source_rule])
        put_obj(self, RuleColor(target_color), transform(target_stem[0]))
        put_obj(self, RuleObject(target_name), transform(target_stem[1]))
        put_obj(self, RuleIs(), transform(target_stem[2]))
        put_rule(self, goal_name, "win", positions=[transform(pos) for pos in goal_rule])
        put_obj(self, RuleObject("baba"), transform(remote_baba))
        put_obj(self, RuleIs(), transform(remote_is))
        put_obj(self, RuleProperty("you"), transform(remote_you))
        for pos in pocket_walls:
            put_obj(self, Wall(), transform(pos))
        for y in range(1, height - 1):
            put_obj(self, Wall(), (width // 2, y))

        put_obj(self, "baba", transform(baba))
        put_obj(self, target, transform(target_pos))
        put_obj(self, target_twin, transform(twin_pos))
        put_obj(self, goal, transform(goal_pos))

        actions = ["down"] + ["down"] * 6 + ["left"]
        if source_on_left:
            actions += ["down"] * 4 + ["left"] * 3
        else:
            actions += ["down"] * 3 + ["left", "down", "left", "left"]
        if flip_vertical:
            actions = self._transform_square_actions(actions, flip_vertical=True)

        self._set_solution(
            f"transfer[you: baba -> {target_color} {target_name}], "
            f"restore[baba is you], break[{target_color} {target_name} is you], goto[{goal_name}]",
            actions,
            variant=variant_idx,
            target=target_name,
            target_color=target_color,
            target_twin_color=twin_color,
            goal=goal_name,
            goal_color=goal_color,
            source_side="left" if source_on_left else "right",
            remote_you_destination=transform(remote_you_dest),
            source_you=transform(source_you),
        )


if __name__ == "__main__":
    # env = make("env/goto_win-no_distractor")
    # env = make("env/make_win-no_distractor_rule")
    # env = make("env/make_win-no_distractor_obj")
    # env = make("env/make_win-no_distractor")
    # env = make("env/make_win-irrelevant_distractor_rule")

    # env = make("env/two_room-goto_win")
    # env = make("env/two_room-make_win")
    # env = make("env/two_room-make_win-no_distractor_obj")
    # env = make("env/two_room-maybe_break_stop-goto_win")

    # env = make("env/two_room-break_stop-make_win-no_distractor_obj")

    # env = make("env/make_win-distr_obj_rule")
    # env = make("env/make_win-distr_obj_rule#no_ball_win")
    # env = make("env/make_win-distr_obj_rule#only_ball_win")

    # env = make("env/two_room-break_stop-make_win-distr_obj-irrelevant_rule")
    # env = make("env/two_room-make_you-make_win")
    env = make("env/two_room-make_wall_win")

    env.reset()
    obs = env.render(mode="matrix")
    print(obs)

    play(env)
