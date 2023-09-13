import os
import pickle
import numpy as np
import copy

from cyclic_boosting import flags


class Logger():
    def __init__(self, save_dir, policy):
        # super().__init__()
        self.id = 1
        self.iter = 0
        self.save_dir = save_dir
        self.model_dir = None
        self.policy = policy
        self.make_dir()
        self.CODs = {}
        self.sorted_CODs = {}
        self.best_features = {}


        # for vote
        self.counter = 0

    def make_dir(self) -> None:
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    def make_model_dir(self) -> None:
        if self.policy == "vote":
            self.model_dir = os.path.join(self.save_dir, f'model_{self.id}')
        elif self.policy == "vote_by_num":
            self.model_dir = os.path.join(self.save_dir, f'model_{self.smallest_id}')
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

    def save_model(self, est, name):
        pickle.dump(est, open(name, 'wb'))

    def save_metrics(self, res, name):
        with open(name, 'w') as f:
            for k, v in res.items():
                f.write(f"[{k}]: {v} \n")

    def save_setting(self, mng, name):
        # to string
        fp = {}
        for f, p in mng.feature_properties.items():
            fp[f] = flags.flags_to_string(p)

        s = {}
        for f, sm in mng.smoothers.items():
            s[f] = sm.name

        with open(name, 'w') as f:
            # feature property
            f.write("=== Feature property ===\n")
            for k, v in fp.items():
                f.write(f"[{k}]: {v} \n")
            f.write("\n")

            # feature
            f.write("=== Feature ===\n")
            f.write(f"{mng.features}\n")
            f.write("\n")

            # smoother
            f.write("=== Explicit Smoother ===\n")
            for k, v in s.items():
                f.write(f"[{k}]: {v} \n")
            f.write("\n")

            # interaction term
            f.write("=== Interaction term ===\n")
            if self.policy == "vote":
                f.write(f"{mng.interaction_term[self.id-1]}\n")
            if self.policy == "vote_by_num":
                f.write(f"{mng.interaction_term[self.smallest_id-1]}\n")

    def compute_COD(self, est, evt, mng):
        print(f"iter: {self.iter+1} / {mng.max_interaction}")
        is_first_iteration = self.iter == 0
        is_last_iteration = mng.max_interaction <= self.iter + 1
        if mng.type == "single":
            self.CODs[est['CB'].feature_groups[0]] = {'COD':evt.result['COD'][self.iter], 'F':evt.result['F'][self.iter]}
            if is_last_iteration:
                self.sorted_CODs = sorted(self.CODs.items(), key=lambda x: x[1]['COD'],reverse=True)
        elif mng.type == "multiple":
            #この中でfeatureを入れるか入れないかを決める
            if is_first_iteration:
                self.best_features = {"best_features": [est['CB'].feature_groups[0]], "best_COD": evt.result['COD'][self.iter]}
                next_features = self.best_features["best_features"]
                next_features.append(list(mng.sorted_features.keys())[self.iter+1])
                mng.get_features(next_features)
            elif not is_last_iteration:
                better = evt.result['COD'][self.iter] > self.best_features["best_COD"]
                if better:
                    self.best_features = {"best_features": mng.features, "best_COD": evt.result['COD'][self.iter]}
                    print('better------------------------------------------------')
                    print(f"best_features{self.best_features}")
                    for keys in evt.result.keys():
                        print(f"{keys}: {evt.result[keys][-1]}", end=", ")
                else:
                    pass
                next_features = copy.deepcopy(self.best_features["best_features"])
                next_features.append(list(mng.sorted_features.keys())[self.iter+1])
                mng.get_features(next_features)
            else:
                pass
        
        self.iter += 1


    
    def reset_count(self):
        self.id = 1
        self.iter = 0

    def log(self, est, evt, mng):
        #self.policyはtrainer.pyの関数run内で定義
        if self.policy == 'vote':
            self.vote(est, evt, mng)
        elif self.policy == 'vote_by_num':
            self.vote_by_smaller_num_of_criteria(est, evt, mng)
        elif self.policy == 'compute_COD':
            self.compute_COD(est, evt, mng)
        self.id += 1