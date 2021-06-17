from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml'],
    validators=[
    ]
)

settings.validators.validate()

assert isinstance(settings.seed, int)
assert isinstance(settings.cores, int)
assert isinstance(settings.env_file, str)
assert isinstance(settings.PPO.epochs_mean_rewards, int)
