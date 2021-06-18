from tools.nets import Activation
from dynaconf import Dynaconf, Validator


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml'],
    validators=[
        Validator("PPO.gamma", gt=0, lt=1),
        Validator("PPO.lam", gt=0, lt=1),
        Validator("PPO.clip_ratio", gte=0.1, lte=0.3),
        Validator("PPO.target_kl", gte=0.01, lte=0.05),
        Validator("ActorCritic.value_lr", gte=0.000001, lte=0.01),
        Validator("ActorCritic.policy_lr", gte=0.0001, lte=0.01),
        Validator("ActorCritic.activation", is_in=Activation.get_values()),
    ]
)

settings.validators.validate()

assert isinstance(settings.seed, int)
assert isinstance(settings.cores, int)
assert isinstance(settings.out_dir, str)
assert isinstance(settings.env_file, str)
assert isinstance(settings.PPO.lam, float)
assert isinstance(settings.save_freq, int)
assert isinstance(settings.PPO.epochs, int)
assert isinstance(settings.PPO.gamma, float)
assert isinstance(settings.PPO.max_ep_len, int)
assert isinstance(settings.PPO.target_kl, float)
assert isinstance(settings.PPO.clip_ratio, float)
assert isinstance(settings.ActorCritic.layers, int)
assert isinstance(settings.PPO.steps_per_epoch, int)
assert isinstance(settings.PPO.env_solved_at, float)
assert isinstance(settings.ActorCritic.value_lr, float)
assert isinstance(settings.ActorCritic.activation, str)
assert isinstance(settings.PPO.epochs_mean_rewards, int)
assert isinstance(settings.ActorCritic.policy_lr, float)
assert isinstance(settings.ActorCritic.hidden_nodes, int)
assert isinstance(settings.ActorCritic.train_value_iters, int)
assert isinstance(settings.ActorCritic.train_policy_iters, int)

