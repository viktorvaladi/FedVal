import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from typing import Optional, Tuple, List
import numpy as np
import math
from ray.util.multiprocessing import Pool
import copy

def multiprocess_evaluate(model, weights, x_test, y_test):
    model.set_weights(weights) 
    preds = model.predict(x_test)
    spec_label_correct_count = [0.0 for i in range(len(y_test[0]))]
    spec_label_all_count = [0.0 for i in range(len(y_test[0]))]
    spec_label_loss_count = [0.0 for i in range(len(y_test[0]))]
    for i in range(len(preds)):
        pred = np.argmax(preds[i])
        true = np.argmax(y_test[i])
        spec_label_all_count[true] = spec_label_all_count[true] +1
        spec_label_loss_count[true] += -(math.log(max(preds[i][true],0.0001)))
        if true == pred:
            spec_label_correct_count[true] = spec_label_correct_count[true] +1
    spec_label_accuracy = []
    spec_label_loss = []
    all_sum = 0
    all_acc_correct = 0
    all_loss_correct = 0
    for i in range(len(spec_label_all_count)):
        all_sum += spec_label_all_count[i]
        spec_label_accuracy.append(spec_label_correct_count[i]/spec_label_all_count[i])
        all_acc_correct += spec_label_correct_count[i]
        spec_label_loss.append(spec_label_loss_count[i]/spec_label_all_count[i])
        all_loss_correct += spec_label_loss_count[i]
    print(f"acc client: {all_acc_correct/all_sum}")
    print(f"spec label client {spec_label_accuracy}")
    # np.mean(spec_label_loss) if we want each label to mean as much, use 
    # all_loss_correct/all_sum instead as first return if you want to promote the distribution of test data
    return np.mean(spec_label_loss), {"accuracy": all_acc_correct/all_sum}, spec_label_accuracy, spec_label_loss

class Poison_detect:
    # s1_factor determines by how much more we want to favor the stronger client updates
    # s2 determines how important it is for labels to not fall behind
    def __init__(self, x_val, y_val, model, s1_overall = 2, s1_label = 3, s2 = 3):
        self.model = model
        self.evclient = Poison_detect.get_eval_fn(self.model, x_val, y_val)
        self.x_test = x_val
        self.y_test = y_val
        self.no_labels = len(y_val[0])
        self.s1_overall = s1_overall
        self.s1_label = s1_label
        self.s2 = s2
        self.pre_reset_s2 = s2
    
    """
    Input is results: List[Tuple]. The list contains Tuples where first element in the tuple is client ID and second 
    element in tuple is a list of ndarrays for the updated client model of client with said ID. last_agg_w is the global 
    models weights for the last round, used to calculate norms. Returns an aggregation of each updated client model based 
    on input parameters
    """
    def calculate_new_aggregated(self,results: List[Tuple], last_agg_w: list):
        label_acc_dict, nodes_acc, loss_dict, label_loss_dict, last_loss, last_label_loss = self.calculate_accs(results)
        adaptives2Loss = []
        adaptives2Parts = []
        weights = []
        # this could be parallelized
        adaptives2Tests = [self.s2, max(1,self.s2-0.5), self.s2+0.5, 3, self.pre_reset_s2]
        i = 0
        for elem in adaptives2Tests:
            self.s2 = elem
            points = {}
            points, overall_mean = self.get_points_overall(loss_dict, results, points=points)
            points = self.get_points_label(label_loss_dict, results, overall_mean, points, last_loss, last_label_loss)
            part_agg = self.points_to_parts(points)
            agg_copy_weights = self.agg_copy_weights(results, part_agg, last_agg_w)
            weights.append(agg_copy_weights)
            loss, acc, _, _ = self.evclient(agg_copy_weights)
            adaptives2Parts.append(part_agg)
            adaptives2Loss.append(loss)
            print(f"acc on {elem}: {acc}")
            i = i+1
        idx_max = np.argmin(adaptives2Loss)
        if idx_max == 3:
            self.pre_reset_s2 = self.s2
        self.s2 = adaptives2Tests[idx_max]
        print(f"self.s2 is now: {self.s2}")
        return weights[idx_max]
    
    def agg_copy_weights(self, results, part_agg, last_weights):
        _, norms_dict = self.calculate_avg_norms1(results,last_weights)
        ret_weights = []
        for elem in norms_dict:
            for i in range(len(norms_dict[elem])):
                if i < len(ret_weights):
                    ret_weights[i] = np.add(ret_weights[i], norms_dict[elem][i]*part_agg[elem])
                else:
                    ret_weights.append(norms_dict[elem][i]*part_agg[elem])
        for i in range(len(ret_weights)):
            ret_weights[i] = np.add(ret_weights[i], last_weights[i])
        return ret_weights
    
    
    def get_norms(self, weights, last_weights):
        norms = []
        for i in range(len(weights)):
            norms.append(np.subtract(weights[i], last_weights[i]))
        return norms
    
    def calculate_avg_norms1(self, results, last_weights):
        norms_dict = {}
        norms_list = []
        for elem in results:
            norm = self.get_norms(elem[1],last_weights)
            norms_dict[elem[0]] = norm
            norms_list.append(norm)
        norms_avg = copy.deepcopy(norms_list[0])
        for w_indx in range(len(norms_list[0])):
            for c_indx in range(1, len(norms_list)):
                norms_avg[w_indx] = np.add(norms_avg[w_indx] , norms_list[c_indx][w_indx])
        for i in range(len(norms_avg)):
            norms_avg[i] = norms_avg[i]/len(norms_list)
        return norms_avg, norms_dict

    def points_to_parts(self, points):
        part_agg = {}
        #make sure no client has negative points
        for elem in points:
            points[elem] = max(0,points[elem])
        sum_points = 0
        for elem in points:
            sum_points += points[elem]
        sum_points = max(000.1, sum_points)
        for elem in points:
            part_agg[elem] = (points[elem] / sum_points)
        return part_agg

    def get_points_overall(self, nodes_acc, results, points = {}):
        #overall points
        # calculate mean absolute deviation for middle 80% of clients
        mean_calc = []
        for elem in nodes_acc:
            mean_calc.append(nodes_acc[elem])
        mean = np.mean(mean_calc)
        all_for_score = []
        for elem in mean_calc:
            #if loss then (mean - elem), if accuracy (mean - elem)
            all_for_score.append(mean - elem)
        mad_calc = all_for_score.copy()
        for i in range(len(mad_calc)):
            mad_calc[i] = abs(mad_calc[i])
        no_elems = round(len(mad_calc))
        mad_calc.sort()
        mad_calc = mad_calc[:no_elems]
        mad = np.mean(mad_calc)
        slope = self.s1_overall/mad
        for i in range(len(all_for_score)):
            points[results[i][0]] = points.get(results[i][0],0) + slope*all_for_score[i] + 10
        #individual label points
        return points, mean
    
    def get_points_label(self, label_acc_dict, results, overall_mean, points):
        #individual label points
        for i in range(self.no_labels):
            mean_calc = []
            for elem in label_acc_dict:
                mean_calc.append(label_acc_dict.get(elem)[i])
            mean = np.mean(mean_calc)
            all_for_score = []
            for elem in mean_calc:
                all_for_score.append(mean - elem)
            mad_calc = all_for_score.copy()
            for j in range(len(mad_calc)):
                mad_calc[j] = abs(mad_calc[j])
            no_elems = round(len(mad_calc))
            mad_calc.sort()
            mad_calc = mad_calc[:no_elems]
            mad = np.mean(mad_calc)
            slope = self.s1_label/mad

            dif = (mean - overall_mean)
            x = ((overall_mean+dif)/overall_mean)
            factor = x**self.s2
            for k in range(len(all_for_score)):
                points[results[k][0]] = points.get(results[k][0],0) + (max(1,factor))*slope*all_for_score[k] + 10
        return points

    def par_results_ev(self, result):
        loss, acc, lab_acc,lab_loss = multiprocess_evaluate(self.data, self.model, result[1], self.x_test, self.y_test)
        return [result[0], loss, acc, lab_acc, lab_loss]

    def calculate_accs(self, results):
        label_acc_dict = {}
        nodes_acc = {}
        loss_dict = {}
        label_loss_dict = {}
        pool = Pool(ray_address="auto")
        evaluated = pool.map(self.par_results_ev, results)
        for elem in evaluated:
            label_acc_dict[elem[0]] = elem[3]
            nodes_acc[elem[0]] = elem[2].get('accuracy')
            loss_dict[elem[0]] = elem[1]
            label_loss_dict[elem[0]] = elem[4]
        #redundant:)
        last_loss = 0
        last_label_loss = 0
        return label_acc_dict, nodes_acc, loss_dict, label_loss_dict, last_loss, last_label_loss

    @staticmethod
    def get_eval_fn(model, x_test,y_test):
        """Return an evaluation function for server-side evaluation."""
        def evaluate(weights) -> Optional[Tuple[float, float]]:
            model.set_weights(weights)
            preds = model.predict(x_test)
            spec_label_correct_count = [0.0 for i in range(len(y_test[0]))]
            spec_label_all_count = [0.0 for i in range(len(y_test[0]))]
            spec_label_loss_count = [0.0 for i in range(len(y_test[0]))]
            for i in range(len(preds)):
                pred = np.argmax(preds[i])
                true = np.argmax(y_test[i])
                spec_label_all_count[true] = spec_label_all_count[true] +1
                spec_label_loss_count[true] += -(math.log(max(preds[i][true],0.0001)))#0.0001 to avoid divide by zero
                if true == pred:
                    spec_label_correct_count[true] = spec_label_correct_count[true] +1
            spec_label_accuracy = []
            spec_label_loss = []
            all_sum = 0
            all_acc_correct = 0
            all_loss_correct = 0
            for i in range(len(spec_label_all_count)):
                all_sum += spec_label_all_count[i]
                spec_label_accuracy.append(spec_label_correct_count[i]/spec_label_all_count[i])
                all_acc_correct += spec_label_correct_count[i]
                spec_label_loss.append(spec_label_loss_count[i]/spec_label_all_count[i])
                all_loss_correct += spec_label_loss_count[i]
            # np.mean(spec_label_loss) if we want each label to mean as much, use 
            # all_loss_correct/all_sum instead as first return if you want to promote the distribution of test data
            return np.mean(spec_label_loss), {"accuracy": all_acc_correct/all_sum}, spec_label_accuracy, spec_label_loss
        return evaluate
