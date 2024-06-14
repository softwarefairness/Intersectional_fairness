class ModelConfigs:
    def __init__(self):
        self.configs = {
            "adult": {
                "feature_extractor": [32],  
                "classifier": [],  
                "sensitive_classifier": []  
            },
            "default": {
                "feature_extractor": [32],
                "classifier": [],
                "sensitive_classifier": []
            },
            "mep1": {
                "feature_extractor": [32],
                "classifier": [],
                "sensitive_classifier": []
            },
            "mep2": {
                "feature_extractor": [32],
                "classifier": [],
                "sensitive_classifier": []
            },
            "german": {
                "feature_extractor": [32],
                "classifier": [],
                "sensitive_classifier": []
            },
            "compas": {
                "feature_extractor": [32],
                "classifier": [],
                "sensitive_classifier": []
            },
        }

    def get_configs(self):
        """
        Return the default model configs.
        
        :return: A dict of model configs
        """
        return self.configs
