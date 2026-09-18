import logging

from pydantic import BaseModel

from vibe.backends.base import ModelPlugin
from vibe.contracts import ALL_CAPABILITY_PROTOCOLS
from vibe.metadata import OutputKind
from vibe.results import ModelResult

logger = logging.getLogger(__name__)


class PluginValidator:
    """Audits plugins for metadata correctness and runtime sync."""

    @classmethod
    def validate_plugin(cls, plugin_cls: type[ModelPlugin], sample_result: ModelResult | None = None) -> bool:
        model_id = plugin_cls.identity.model_id if getattr(plugin_cls, "identity", None) else plugin_cls.__name__
        profile = getattr(plugin_cls, "profile", None)
        success = True

        def _fail(msg: str) -> None:
            nonlocal success
            success = False
            logger.error("[%s] VALIDATION FAILED: %s", model_id, msg)

        if not profile:
            _fail("Plugin is missing a 'profile' declaration.")
            return False

        out_spec = profile.output_spec

        if out_spec.kind == OutputKind.SCORE and out_spec.entry_extras:
            _fail("Model outputs SCORE but declared entry_extras. Use output_extras instead.")

        for s in plugin_cls.settings:
            if not s.id:
                _fail("Encountered SettingGroupSpec with empty ID.")
            if not (isinstance(s.model_type, type) and issubclass(s.model_type, BaseModel)):
                _fail(f"Setting '{s.id}' model_type must be a subclass of pydantic.BaseModel.")
            try:
                s.json_schema()
            except Exception as exc:
                _fail(f"Setting '{s.id}' failed to generate JSON schema: {exc}")

        implements = getattr(plugin_cls, "implements", ())
        if not isinstance(implements, tuple):
            _fail("'implements' must be a tuple of Protocol types.")
        else:
            for proto in implements:
                if proto not in ALL_CAPABILITY_PROTOCOLS:
                    _fail(
                        f"Declared protocol {getattr(proto, '__name__', str(proto))} is not a known capability protocol."
                    )

        if sample_result is not None and sample_result.output_type != out_spec.kind:
            _fail(f"Declared output kind '{out_spec.kind.value}', but returned '{sample_result.output_type.value}'.")

        if success:
            logger.info("[%s] Validation passed.", model_id)

        return success
