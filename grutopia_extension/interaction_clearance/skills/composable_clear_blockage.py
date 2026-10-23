def run(ctx):
    outcome = ctx.execute('base_move', target='grasp')
    if not outcome.success:
        return outcome
    outcome = ctx.execute('arm_plan', mode='grasp')
    if not outcome.success:
        return outcome
    outcome = ctx.execute('grasp_lift')
    if not outcome.success:
        return outcome
    outcome = ctx.execute('base_move', target='placement')
    if not outcome.success:
        return outcome
    outcome = ctx.execute('arm_plan', mode='place')
    if not outcome.success:
        return outcome
    outcome = ctx.execute('place_release')
    if not outcome.success:
        return outcome
    if ctx.task.return_to_start:
        outcome = ctx.execute('base_move', target='home')
        if not outcome.success:
            return outcome
    return ctx.verify_clearance()
