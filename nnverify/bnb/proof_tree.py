from nnverify.bnb import branch

from nnverify.common import Status
from nnverify.specs.spec import SpecList, Spec


class ProofTree:
    def __init__(self, root):
        self.root = root

    def get_leaves(self):
        if self.root is None:
            raise ValueError("Proof Tree root is not set!")
        leaves = SpecList()
        queue = [self.root]

        while len(queue) != 0:
            nd = queue.pop()

            if len(nd.children) == 0:
                leaves.append(nd)
            else:
                for child in nd.children:
                    queue.append(child)

        # If a spec has adv ex status we move it first
        adv_ex_id = None
        for i in range(len(leaves)):
            if leaves[i].status == Status.ADV_EXAMPLE:
                adv_ex_id = i
            # reset the spec status
            leaves[i].reset_status()

        if adv_ex_id is not None:
            leaves.insert(0, leaves.pop(adv_ex_id))

        return leaves

    # Prune splits that have score <= threshold
    def get_pruned_leaves(self, threshold, split_type):
        if self.root is None:
            raise ValueError("Proof Tree root is not set!")

        new_proof_tree, _ = self.get_pruned_tree(threshold, split_type)
        return new_proof_tree.get_leaves()

    def get_pruned_tree(self, threshold, split_type):
        new_root = Spec(self.root.input_spec, relu_spec=self.root.relu_spec)
        new_proof_tree = ProofTree(new_root)

        old_node_to_new_node_map = {self.root: new_root}
        # Create a new tree that only computes strong splits
        queue = [self.root]
        while len(queue) != 0:
            old_nd = queue.pop()

            if old_nd not in old_node_to_new_node_map or len(old_nd.children) == 0:
                continue

            new_nd = old_node_to_new_node_map[old_nd]

            worst_case_improvement = old_nd.children[0].lb - old_nd.lb
            worst_case_child = 0

            for i in range(len(old_nd.children)):
                if old_nd.children[i].lb - old_nd.lb < worst_case_improvement:
                    worst_case_improvement = old_nd.children[i].lb - old_nd.lb
                    worst_case_child = i

            if worst_case_improvement < threshold:
                old_node_to_new_node_map[old_nd.children[worst_case_child]] = new_nd
            else:
                chosen_split = old_nd.chosen_split
                new_children = branch.split_chosen_spec(new_nd, split_type, chosen_split)
                for i in range(len(old_nd.children)):
                    old_node_to_new_node_map[old_nd.children[i]] = new_children[i]

            for child in old_nd.children:
                queue.append(child)
        return new_proof_tree, old_node_to_new_node_map

    def compute_subtree_size(self):
        post_order = self.get_preorder()
        post_order.reverse()
        for nd in post_order:
            nd.subtree_size = 0
            for child in nd.children:
                nd.subtree_size += child.subtree_size

    def get_node_imp_scores(self):
        post_order = self.get_preorder()
        post_order.reverse()
        imp_scores = {}
        for nd in post_order:
            if len(nd.children) == 0:
                imp_scores[nd] = 0
                continue

            min_improvement = 1e5
            for child in nd.children:
                min_improvement = min(min_improvement, child.ld - nd.lb)
            imp_scores[nd] = min_improvement
        return imp_scores

    def get_preorder(self):
        preorder = []
        queue = [self.root]
        while len(queue) != 0:
            nd = queue.pop()
            preorder.append(nd)
            for child in nd.children:
                queue.append(child)
        return preorder

    def get_observed_split_score(self):
        queue = [self.root]
        total_score = {}
        split_count = {}
        observed_score = {}

        while len(queue) != 0:
            nd = queue.pop()

            if len(nd.children) == 0:
                continue

            worst_case_improvement = nd.children[0].lb - nd.lb

            for child in nd.children:
                if child.lb - nd.lb < worst_case_improvement:
                    worst_case_improvement = child.lb - nd.lb

            if nd.chosen_split not in total_score:
                total_score[nd.chosen_split] = 0
                split_count[nd.chosen_split] = 0

            total_score[nd.chosen_split] += worst_case_improvement
            split_count[nd.chosen_split] += 1

            for child in nd.children:
                queue.append(child)

        for k, v in total_score.items():
            observed_score[k] = total_score[k] / split_count[k]

        return observed_score

    def get_best_observed_split(self, new_nd, obs_scores, backup_split):
        cur_nd = new_nd
        done_splits = {}
        while cur_nd is not None:
            done_splits[cur_nd.chosen_split] = True
            cur_nd = cur_nd.parent

        best_split_score = obs_scores[backup_split]
        best_split = backup_split

        for split, score in obs_scores.items():
            if score > best_split_score and (split not in done_splits):
                best_split_score = score
                best_split = split

        return best_split
