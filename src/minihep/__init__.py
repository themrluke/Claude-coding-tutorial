"""minihep: a small re-implementation of hepattn, built up week by week.

Every module here mirrors a module in hepattn (same names, smaller scope), so once
you have written the minihep version you can read the real one line by line:

    minihep.data.dataset        <-> hepattn.experiments.trackml.data
    minihep.models.attention    <-> hepattn.models.attention
    minihep.models.encoder      <-> hepattn.models.encoder
    minihep.models.decoder      <-> hepattn.models.decoder
    minihep.models.maskformer   <-> hepattn.models.maskformer
    minihep.models.matcher      <-> hepattn.models.matcher
    minihep.models.tasks        <-> hepattn.models.task
    minihep.lightning.wrapper   <-> hepattn.models.wrapper
    minihep.cli                 <-> hepattn.utils.cli
"""
