"""Comet logger wrapper (provided). Mirror of src/hepattn/utils/loggers.py.

Lightning 2.5's CometLogger takes most options through ``**kwargs``. jsonargparse cannot check
or set keys it cannot see in a signature, so the CLI could not fill in ``offline_directory`` or
``name``. Spelling them out as real parameters fixes that.

Comet needs an API key. Put it in ``~/.comet.config``::

    [comet]
    api_key = <your key>
    workspace = themrluke

or export ``COMET_API_KEY``. Never commit the key. ``online: false`` writes an offline archive
into ``offline_directory`` that you can upload later with ``comet upload <file>.zip``.
"""

from lightning.pytorch.loggers import CometLogger as _CometLogger


class CometLogger(_CometLogger):
    def __init__(
        self,
        name: str | None = None,
        project: str | None = None,
        workspace: str | None = None,
        offline_directory: str | None = None,
        online: bool | None = None,
        experiment_key: str | None = None,
        mode: str | None = None,
        log_env_details: bool = True,
    ):
        super().__init__(
            name=name,
            project=project,
            workspace=workspace,
            offline_directory=offline_directory,
            online=online,
            experiment_key=experiment_key,
            mode=mode,
            log_env_details=log_env_details,
        )
