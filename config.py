from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml'],
    validators=[
    ]
)

settings.validators.validate()

assert isinstance(settings.env_file, str)