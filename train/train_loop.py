import torch
import time
import numpy as np
from  config import Config as cfg

class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self):
        pass

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        # logger = logging.getLogger(__name__)
        # logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        for self.iter in range(start_iter, max_iter):
            self.run_step()
        # with EventStorage(start_iter) as self.storage:
        #     try:
        #
        #         for self.iter in range(start_iter, max_iter):
        #             self.before_step()
        #             self.run_step()
        #             self.after_step()
        #     finally:
        #         self.after_train()

    def run_step(self):
        raise NotImplementedError

#
# class SimpleTrainer(TrainerBase):
#     """
#     A simple trainer for the most common type of task:
#     single-cost single-optimizer single-data-source iterative optimization.
#     It assumes that every step, you:
#
#     1. Compute the loss with a data from the data_loader.
#     2. Compute the gradients with the above loss.
#     3. Update the model with the optimizer.
#
#     If you want to do anything fancier than this,
#     either subclass TrainerBase and implement your own `run_step`,
#     or write your own training loop.
#     """
#
#     def __init__(self, model, data_loader, optimizer):
#         """
#         Args:
#             model: a torch Module. Takes a data from data_loader and returns a
#                 dict of losses.
#             data_loader: an iterable. Contains data to be used to call model.
#             optimizer: a torch optimizer.
#         """
#         super().__init__()
#
#         """
#         We set the model to training mode in the trainer.
#         However it's valid to train a model that's in eval mode.
#         If you want your model (or a submodule of it) to behave
#         like evaluation during training, you can overwrite its train() method.
#         """
#         model.train()
#
#         self.model = model
#         self.data_loader = data_loader
#         self._data_loader_iter = iter(data_loader)
#         self.optimizer = optimizer
#
#     def run_step(self):
#         """
#         Implement the standard training logic described above.
#         """
#         assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
#         start = time.perf_counter()
#         """
#         If your want to do something with the data, you can wrap the dataloader.
#         """
#         data = next(self._data_loader_iter)
#         data_time = time.perf_counter() - start
#
#         """
#         If your want to do something with the losses, you can wrap the model.
#         """
#         loss_dict = self.model(data)
#         losses = sum(loss for loss in loss_dict.values())
#         self._detect_anomaly(losses, loss_dict)
#
#         metrics_dict = loss_dict
#         metrics_dict["data_time"] = data_time
#         self._write_metrics(metrics_dict)
#
#         """
#         If you need accumulate gradients or something similar, you can
#         wrap the optimizer with your custom `zero_grad()` method.
#         """
#         self.optimizer.zero_grad()
#         losses.backward()
#
#         """
#         If you need gradient clipping/scaling or other processing, you can
#         wrap the optimizer with your custom `step()` method.
#         """
#         self.optimizer.step()
#
#     def _detect_anomaly(self, losses, loss_dict):
#         if not torch.isfinite(losses).all():
#             raise FloatingPointError(
#                 "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
#                     self.iter, loss_dict
#                 )
#             )
#
#     def _write_metrics(self, metrics_dict: dict):
#         """
#         Args:
#             metrics_dict (dict): dict of scalar metrics
#         """
#         metrics_dict = {
#             k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
#             for k, v in metrics_dict.items()
#         }
#         # gather metrics among all workers for logging
#         # This assumes we do DDP-style training, which is currently the only
#         # supported method in detectron2.
#         all_metrics_dict = comm.gather(metrics_dict)
#
#         if comm.is_main_process():
#             if "data_time" in all_metrics_dict[0]:
#                 # data_time among workers can have high variance. The actual latency
#                 # caused by data_time is the maximum among workers.
#                 data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
#                 self.storage.put_scalar("data_time", data_time)
#
#             # average the rest metrics
#             metrics_dict = {
#                 k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
#             }
#             total_losses_reduced = sum(loss for loss in metrics_dict.values())
#
#             self.storage.put_scalar("total_loss", total_losses_reduced)
#             if len(metrics_dict) > 1:
#                 self.storage.put_scalars(**metrics_dict)

# class DefaultTrainer(SimpleTrainer):
#     """
#     A trainer with default training logic. Compared to `SimpleTrainer`, it
#     contains the following logic in addition:
#
#     1. Create model, optimizer, scheduler, dataloader from the given config.
#     2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists.
#     3. Register a few common hooks.
#
#     It is created to simplify the **standard model training workflow** and reduce code boilerplate
#     for users who only need the standard training workflow, with standard features.
#     It means this class makes *many assumptions* about your training logic that
#     may easily become invalid in a new research. In fact, any assumptions beyond those made in the
#     :class:`SimpleTrainer` are too much for research.
#
#     The code of this class has been annotated about restrictive assumptions it mades.
#     When they do not work for you, you're encouraged to write your own training logic.
#
#     Also note that the behavior of this class, like other functions/classes in
#     this file, is not stable, since it is meant to represent the "common default behavior".
#     It is only guaranteed to work well with the standard models and training workflow in detectron2.
#     To obtain more stable behavior, write your own training logic with other public APIs.
#
#     Attributes:
#         scheduler:
#         checkpointer (DetectionCheckpointer):
#         cfg (CfgNode):
#     """
#
#     def __init__(self, model , data_loader, optimizer):
#         """
#         Args:
#             cfg (CfgNode):
#         """
#         # Assume these objects must be constructed in this order.
#         model =model
#         # if cfg.FREEZE_ALL:
#         #     # Freeze model parameters except for viewpoint
#         #     for name,param in model.named_parameters():
#         #         if param.requires_grad and 'viewpoint' not in name:
#         #             param.requires_grad = False
#         optimizer = optimizer
#         data_loader = data_loader
#         super().__init__(model, data_loader, optimizer)
#
#         # self.scheduler = self.build_lr_scheduler(cfg, optimizer)
#         # Assume no other objects need to be checkpointed.
#         # We can later make it checkpoint the stateful hooks
#         # self.checkpointer = DetectionCheckpointer(
#         #     # Assume you want to save checkpoints together with logs/statistics
#         #     model,
#         #     cfg.OUTPUT_DIR,
#         #     optimizer=optimizer,
#         #     scheduler=self.scheduler,
#         # )
#         self.start_iter = 0
#         self.max_iter = cfg.Model.Output.MAX_ITER
#         self.cfg = cfg
#
#         # self.register_hooks(self.build_hooks())
#
#     def resume_or_load(self, resume=True):
#         """
#         If `resume==True`, and last checkpoint exists, resume from it.
#
#         Otherwise, load a model specified by the config.
#
#         Args:
#             resume (bool): whether to do resume or not
#         """
#         # The checkpoint stores the training iteration that just finished, thus we start
#         # at the next iteration (or iter zero if there's no checkpoint).
#         self.checkpointer._set_resume(resume)
#         self.start_iter = (
#             self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume).get(
#                 "iteration", -1
#             )
#             + 1
#         )
#
#
#     def build_hooks(self):
#         """
#         Build a list of default hooks.
#
#         Returns:
#             list[HookBase]:
#         """
#         cfg = self.cfg.clone()
#         cfg.defrost()
#         cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
#
#         ret = [
#             hooks.IterationTimer(),
#             hooks.LRScheduler(self.optimizer, self.scheduler),
#             hooks.PreciseBN(
#                 # Run at the same freq as (but before) evaluation.
#                 cfg.TEST.EVAL_PERIOD,
#                 self.model,
#                 # Build a new data loader to not affect training
#                 self.build_train_loader(cfg),
#                 cfg.TEST.PRECISE_BN.NUM_ITER,
#             )
#             if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
#             else None,
#         ]
#
#         # Do PreciseBN before checkpointer, because it updates the model and need to
#         # be saved by checkpointer.
#         # This is not always the best: if checkpointing has a different frequency,
#         # some checkpoints may have more precise statistics than others.
#         if comm.is_main_process():
#             ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
#
#         def test_and_save_results():
#             self._last_eval_results = self.test(self.cfg, self.model)
#             return self._last_eval_results
#
#         # Do evaluation after checkpointer, because then if it fails,
#         # we can use the saved checkpoint to debug.
#         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
#
#         if comm.is_main_process():
#             # run writers in the end, so that evaluation metrics are written
#             ret.append(hooks.PeriodicWriter(self.build_writers()))
#         return ret
#
#     def build_writers(self):
#         """
#         Build a list of default writers, that write metrics to the screen,
#         a json file, and a tensorboard event file respectively.
#
#         Returns:
#             list[Writer]: a list of objects that have a ``.write`` method.
#         """
#         # Assume the default print/log frequency.
#         return [
#             # It may not always print what you want to see, since it prints "common" metrics only.
#             CommonMetricPrinter(self.max_iter),
#             JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
#             TensorboardXWriter(self.cfg.OUTPUT_DIR),
#         ]
#
#     def train(self):
#         """
#         Run training.
#
#         Returns:
#             OrderedDict of results, if evaluation is enabled. Otherwise None.
#         """
#         super().train(self.start_iter, self.max_iter)
#         if hasattr(self, "_last_eval_results") and comm.is_main_process():
#             verify_results(self.cfg, self._last_eval_results)
#             return self._last_eval_results
#
#
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name):
#         """
#         Returns:
#             DatasetEvaluator
#         """
#         raise NotImplementedError
#
#     @classmethod
#     def test(cls, cfg, model, evaluators=None):
#         """
#         Args:
#             cfg (CfgNode):
#             model (nn.Module):
#             evaluators (list[DatasetEvaluator] or None): if None, will call
#                 :meth:`build_evaluator`. Otherwise, must have the same length as
#                 `cfg.DATASETS.TEST`.
#
#         Returns:
#             dict: a dict of result metrics
#         """
#         logger = logging.getLogger(__name__)
#         if isinstance(evaluators, DatasetEvaluator):
#             evaluators = [evaluators]
#         if evaluators is not None:
#             assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
#                 len(cfg.DATASETS.TEST), len(evaluators)
#             )
#
#         results = OrderedDict()
#         for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
#             data_loader = cls.build_test_loader(cfg, dataset_name)
#             # When evaluators are passed in as arguments,
#             # implicitly assume that evaluators can be created before data_loader.
#             evaluator = (
#                 evaluators[idx]
#                 if evaluators is not None
#                 else cls.build_evaluator(cfg, dataset_name)
#             )
#             results_i = inference_on_dataset(model, data_loader, evaluator)
#             results[dataset_name] = results_i
#             if comm.is_main_process():
#                 assert isinstance(
#                     results_i, dict
#                 ), "Evaluator must return a dict on the main process. Got {} instead.".format(
#                     results_i
#                 )
#                 logger.info("Evaluation results for {} in csv format:".format(dataset_name))
#                 print_csv_format(results_i)
#
#         if len(results) == 1:
#             results = list(results.values())[0]
#         return results
