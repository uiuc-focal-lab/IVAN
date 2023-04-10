from nnverify.common import Domain


class TrainArgs:
    def __init__(self, lr, epochs, schedule_length=0, warmup_lr=None, warmup_epochs=0, gpu="0", trainer=Domain.BASE,
                 val_method='ibp', batch_size=100, scaled_score_init=False, resume=False, seed=0, no_cuda=False, freeze_bn=False,
                 scores_init_type=None, scale_rand_init=False, is_semisup=False, optimizer='sgd', momentum=0.9, wd=1e-4,
                 lr_schedule='cosine', snip_init=False, evaluate=False, epsilon=0.001, configs="./configs/configs.yml", mode='prune',
                 ):
        """
        TODO WIP: This is copied from https://github.com/inspire-group/hydra. It has many unnecessary things that can be deleted over time

        args.trainer,
        lr: learning rate,
        epochs
        warmup_lr
        warmup_epochs
        schedule_length 1
        batch-size 128
        trainer: training domain
        val_method: validation domain ("base", "adv", "mixtrain", "ibp", "smooth", "freeadv")
        scores_init_type: "kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal"
        optimizer: "sgd", "adam", "rmsprop"
        wd: weight decay
        lr_schedule: "step", "cosine"
        TODO: Change this to global property
        epsilon: local robustness epsilon
        """
        self.lr = lr
        self.schedule_length = schedule_length
        self.warmup_lr = warmup_lr
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.trainer = trainer
        self.val_method = val_method
        self.batch_size = batch_size
        self.scaled_score_init = scaled_score_init
        self.gpu = gpu
        self.resume = resume
        self.seed = seed
        self.no_cuda = no_cuda
        self.freeze_bn = freeze_bn
        self.scores_init_type = scores_init_type
        self.is_semisup = is_semisup
        self.scale_rand_init = scale_rand_init
        self.snip_init = snip_init
        self.evaluate = evaluate
        self.epsilon = epsilon

        # Training
        self.optimizer = optimizer
        self.momentum = momentum
        self.wd = wd
        self.lr_schedule = lr_schedule

        # Constants
        self.result_dir = "logs"
        self.exp_name = "trial"

        self.configs = configs

        self.source_net = None
        self.start_epoch = 0
        self.mode = mode
        self.print_freq = 100
