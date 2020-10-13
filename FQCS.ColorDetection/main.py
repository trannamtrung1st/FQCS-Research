from FQCS.detector import FQCSDetector
import json

def main():
    # init
    detector = FQCSDetector()
    
    # save config
    detector.save_config("config.json")
    
    # load config
    detector.load_config("config.json")
    c_dict = detector.config.get_dict()
    print(json.dumps(c_dict, indent=2))


if __name__ == "__main__":
    main()