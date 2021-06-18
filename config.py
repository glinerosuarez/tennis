from tools.nets import Activation
from dynaconf import Dynaconf, Validator


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml'],
    validators=[
        Validator("ActorCritic.value_lr", gte=0.0001, lte=0.01),
        Validator("ActorCritic.policy_lr", gte=0.0001, lte=0.01),
        Validator("ActorCritic.activation", is_in=Activation.get_values()),
    ]
)

settings.validators.validate()

assert isinstance(settings.seed, int)
assert isinstance(settings.cores, int)
assert isinstance(settings.env_file, str)
assert isinstance(settings.ActorCritic.layers, int)
assert isinstance(settings.ActorCritic.value_lr, float)
assert isinstance(settings.ActorCritic.activation, str)
assert isinstance(settings.PPO.epochs_mean_rewards, int)
assert isinstance(settings.ActorCritic.policy_lr, float)
assert isinstance(settings.ActorCritic.hidden_nodes, int)
assert isinstance(settings.ActorCritic.train_value_iters, int)
assert isinstance(settings.ActorCritic.train_policy_iters, int)

