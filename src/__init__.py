from gym.envs.registration import register

register(
    id="FbsEnv-v0",
    entry_point="src.utils.DataExtractor:DataProcessingEnv",
)
