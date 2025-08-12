# --- maml_trainer.py (完整替换文件) ---

import torch
import torch.optim as optim
from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_

class MAMLTrainer:
    def __init__(self, model, loss_fn, inner_lr, meta_optimizer, num_inner_steps, device, **kwargs):
        self.model = model
        self.loss_fn = loss_fn
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.device = device
        self.model.to(self.device)
        self.meta_optimizer = meta_optimizer
        
        print(f"MAMLTrainer (Final Canonical Version) Initialized with custom optimizer.")

    def outer_loop_batch(self, batched_task_data):
        self.meta_optimizer.zero_grad()
        
        support_images = batched_task_data['support_images'].to(self.device)
        support_masks = batched_task_data['support_masks'].to(self.device)
        query_images = batched_task_data['query_images'].to(self.device)
        query_masks = batched_task_data['query_masks'].to(self.device)

        num_tasks_in_batch = support_images.size(0)
        total_query_loss = torch.tensor(0., device=self.device)

        for i in range(num_tasks_in_batch):
            fast_weights = OrderedDict(self.model.named_parameters())

            for step in range(self.num_inner_steps):
                support_logits = self.model(support_images[i], params=fast_weights)
                inner_loss = self.loss_fn(support_logits, support_masks[i], reduction='mean')
                
                # --- 核心修改在这里 ---
                grads = torch.autograd.grad(
                    inner_loss, 
                    fast_weights.values(), 
                    create_graph=False,
                    allow_unused=True  # <--- 添加这个参数！
                )
                
                # 更新快速权重
                fast_weights = OrderedDict(
                    # 当 grad 为 None 时，参数不更新
                    (name, param - self.inner_lr * grad if grad is not None else param)
                    for ((name, param), grad) in zip(fast_weights.items(), grads)
                )

            query_logits = self.model(query_images[i], params=fast_weights)
            query_loss = self.loss_fn(query_logits, query_masks[i], reduction='mean')
            
            total_query_loss += query_loss

        average_query_loss = total_query_loss / num_tasks_in_batch
        average_query_loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.meta_optimizer.step()

        return average_query_loss.item()