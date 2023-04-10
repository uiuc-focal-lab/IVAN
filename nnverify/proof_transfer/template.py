class TemplateStore:
    def __init__(self):
        self.template_map = {}

    def get_template(self, prop):
        if prop in self.template_map:
            return self.template_map[prop]
        return None

    def add_tree(self, prop, root):
        if prop not in self.template_map:
            self.template_map[prop] = Template()
        self.template_map[prop].set_tree(root)

    def get_leaf_nodes(self, prop):
        if prop not in self.template_map:
            raise ValueError("We expect that the leaf nodes are available.")
        assert self.template_map[prop].proof_tree is not None
        return self.template_map[prop].proof_tree.get_leaves()

    def get_proof_tree(self, prop):
        if prop not in self.template_map:
            raise ValueError("We expect that the leaf nodes are available.")
        return self.template_map[prop].proof_tree

    # Remove this as split scores will also be part of proof tree now
    def add_split_scores(self, prop, observed_split_score):
        if prop not in self.template_map:
            self.template_map[prop] = Template()
        self.template_map[prop].relu_score = observed_split_score

    def get_split_score(self, prop, chosen_split):
        if prop in self.template_map and chosen_split in self.template_map[prop].relu_score:
            return self.template_map[prop].relu_score[chosen_split]
        return None

    def is_tree_available(self, prop):
        if prop in self.template_map and self.template_map[prop].proof_tree is not None:
            return True
        return False


class Template:
    """
    @field final_specs: this captures the final split nodes (leaves of the binary tree) that were not split further
        during the verification

    @field relu_score: this captures the effectiveness of relu split
    """
    def __init__(self):
        self.relu_score = {}
        self.proof_tree = None

    def set_tree(self, root):
        self.proof_tree = root

    def get_split_score(self, relu):
        if relu in self.relu_score:
            return self.relu_score[relu]
        return None

