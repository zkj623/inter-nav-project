def run(ctx):
    outcome = ctx.execute('base_move', target='grasp')
    if not outcome.success:
        return outcome
    for attempt in range(3):
        outcome = ctx.execute('arm_plan', mode='grasp')
        if outcome.success:
            break
        if ctx.task.object_kind != 'stool' or outcome.reason != 'ARM_PLAN_REJECTED' or attempt == 2:
            return outcome
        outcome = ctx.execute('base_move', target='grasp', grasp_distance_offset_m=-0.01 * (attempt + 1))
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
