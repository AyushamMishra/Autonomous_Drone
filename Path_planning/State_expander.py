from Path_planning.state import State

class StateExpander:
    def __init__(self, motion_primitives, transition_model):
        self.motion_primitives = motion_primitives
        self.tm = transition_model

    def expand(self, state: State):
        successors = []

        for primitive in self.motion_primitives.primitives():

            result = self.tm.propagate(state, primitive)

            if result is None:
                continue

            nx, ny, nyaw, cost = result

            successors.append(
                State(
                    x=nx,
                    y=ny,
                    yaw=nyaw,
                    g=state.g + cost
                )
            )

        return successors