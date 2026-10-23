def run(ctx):
    ctx.report('Upstream requires interaction: move the designated box to its allowed placement target.')
    outcome = ctx.execute('pick_place', object_id=ctx.task.object_id, place_xy=ctx.task.place_xy)
    if not outcome.success:
        return outcome
    return ctx.verify_clearance()
