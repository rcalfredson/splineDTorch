import glob
import json
import os
import random


class NetRetrainManager:
    def __init__(self, nets_dir):
        self.nets_dir = nets_dir
        self.tracker_filename = os.path.join(self.nets_dir, "retrain_tracker.json")
        self.net_retrain_statuses = {
            os.path.basename(n): False
            for n in glob.glob(os.path.join(nets_dir, "*.pth"))
        }

    def get_random_net_to_retrain(self, debug=False):
        self.sync_net_statuses_from_file()

        net_to_train = random.choice(
            [
                k
                for k in self.net_retrain_statuses
                if self.net_retrain_statuses[k] == False
            ]
        )

        self.sync_net_statuses_from_file()
        if not debug:
            self.net_retrain_statuses[net_to_train] = True
        self.update_tracker_file()
        return net_to_train

    def nets_remaining_to_retrain(self) -> bool:
        self.sync_net_statuses_from_file()
        return False in self.net_retrain_statuses.values()

    def sync_net_statuses_from_file(self):
        if os.path.isfile(self.tracker_filename):
            with open(self.tracker_filename, "r") as f:
                self.net_retrain_statuses = json.load(f)
        else:
            self.update_tracker_file()

    def update_tracker_file(self):
        with open(self.tracker_filename, "w") as f:
            json.dump(self.net_retrain_statuses, f, ensure_ascii=False, indent=4)
