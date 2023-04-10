from nnverify.common import Status


class Reluspec:
    def __init__(self, relu_mask):
        self.relu_mask = relu_mask
        self.status = Status.UNKNOWN

    def split_spec(self, split, chosen_relu_id):
        relu_mask = self.relu_mask

        relu_mask1 = {}
        relu_mask2 = {}

        for relu in relu_mask.keys():
            if relu == chosen_relu_id:
                relu_mask1[relu] = -1
                relu_mask2[relu] = 1
            else:
                relu_mask1[relu] = relu_mask[relu]
                relu_mask2[relu] = relu_mask[relu]

        return [Reluspec(relu_mask1), Reluspec(relu_mask2)]
