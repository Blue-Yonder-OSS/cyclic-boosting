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

    def vote(self, est, evt, mng):
        is_first_iteration = len(self.log_data) <= 0
        if is_first_iteration:
            for metrics, value in evt.result.items():
                self.log_data[metrics] = np.nanmean(value)

            self.make_model_dir()
            self.save_metrics(self.log_data,
                              os.path.join(self.model_dir,
                                           f'metrics_{self.id}.txt'))
            self.save_setting(mng,
                              os.path.join(self.model_dir,
                                           f'setting_{self.id}.txt'))
            self.save_model(est,
                            os.path.join(self.model_dir,
                                         f'model_{self.id}.pkl'))
        else:
            # NOTE: 小さい値を持つほうがよい指標ばかりであるのが前提
            # 修正の必要の可能性あり
            cnt = 0
            report = {}
            for metrics, value in evt.result.items():
                before = self.log_data[metrics]
                after = np.nanmean(value)
                if abs(before) > abs(after):
                    cnt += 1
                report[metrics] = after

            print(self.counter, cnt)
            if self.counter < cnt:
                print(self.log_data)
                print(report)
                print(self.counter, cnt)
                # NOTE: 投票ロジックを賢くする
                self.make_model_dir()
                self.save_metrics(self.log_data,
                                  os.path.join(self.model_dir,
                                               f'metrics_{self.id}.txt'))
                self.save_setting(mng,
                                  os.path.join(self.model_dir,
                                               f'setting_{self.id}.txt'))
                self.save_model(est,
                                os.path.join(self.model_dir,
                                             f'model_{self.id}.pkl'))
                self.counter = cnt
                self.log_data = report

    def compute_COD(self, est, evt, mng):
        print(f"iter: {self.iter} / {mng.max_interaction-1}")
        is_first_iteration = self.iter == 0
        is_last_iteration = mng.max_interaction-1 <= self.iter
        if mng.type == "single":
            self.CODs[est['CB'].feature_groups[0]] = {'COD':evt.result['COD'][self.iter], 'F':evt.result['F'][self.iter]}
            if is_last_iteration:
                self.sorted_CODs = sorted(self.CODs.items(), key=lambda x: x[1]['COD'], reverse=True)
        elif mng.type == "multiple":
            #この中でfeatureを入れるか入れないかを決める
            if is_first_iteration:
                self.best_features = {"best_features": [est['CB'].feature_groups[0]], "best_COD": evt.result['COD'][self.iter]}
                next_features = copy.deepcopy(self.best_features["best_features"])
                next_features.append(mng.sorted_features[self.iter])
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
                next_features.append(mng.sorted_features[self.iter])
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