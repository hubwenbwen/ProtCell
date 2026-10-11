from finetuner.BaseFineTuner import BaseFineTuner, unwrap_model
from model.model import ProteinLanguageModel
import os
import torch

class AddNewProteinFineTuner(BaseFineTuner):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.old_vocab_size = None
        self.added = None
        
    def add_new_proteins(self):
        self.added, self.old_vocab_size = self.model.add_new_proteins(self.vars_names)
        print(f"Added {self.added} new proteins to the tokenizer. Old vocab size: {self.old_vocab_size}, New vocab size: {self.model.tokenizer.vocab_size}")

    def deploy_lora(self,
                    lora_rank: int = 8,
                    lora_alpha: int = 16,
                    lora_dropout: float = 0.05,):
        
        from peft import get_peft_model, LoraConfig, TaskType
        # Step 0: unwrap core
        core = unwrap_model(self.model)

        # Step 1: Inject LoRA first
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["out_proj", "linear1", "linear2"],
            task_type=TaskType.FEATURE_EXTRACTION,
            bias="none"
        )
        core.decoder = get_peft_model(core.decoder, lora_config)

        # Step 2: Freeze all parameters
        for name, param in core.named_parameters():
            param.requires_grad = False

        # Step 3: Unfreeze LoRA parameters
        for name, param in core.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        # Step 4: Unfreeze embedding & cls head if needed
        core.embedding.protein_embed.embedding.weight.requires_grad = True
        # 注册 grad_mask_hook
        embed_weight = core.embedding.protein_embed.embedding.weight
        def grad_mask_hook(grad):
            grad[:self.old_vocab_size] = 0.0
            return grad
        embed_weight.register_hook(grad_mask_hook)

        for name, param in core.named_parameters():
            if "cls_global_head" in name or "cls_proj" in name:
                param.requires_grad = True
 
    def on_fit_start(self, **kwargs):
        self.add_new_proteins()
        self.model.train()
        self.prepare_dataloader()
        
        lora_rank = kwargs.get("lora_rank", 8)
        lora_alpha = kwargs.get("lora_alpha", 16)
        lora_dropout = kwargs.get("lora_dropout", 0.05)
        self.deploy_lora(lora_rank, lora_alpha, lora_dropout)
        
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler(T_max=10)
       
    def save_model(self, save_dir: str = None, file_name: str = None, epoch: int = 0, step: int = 0, **kwargs):
        import copy
        core = unwrap_model(self.model)
        core_copy = copy.deepcopy(core)
        if self.expr_name:
            save_dir = os.path.join(save_dir, self.expr_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # 在副本上合并LoRA
        core_copy.decoder = core_copy.decoder.merge_and_unload()
        
        # 使用副本的save_model方法保存
        save_path = os.path.join(save_dir, file_name)
        core_copy.save_model(save_path)
        
        # 删除副本释放内存
        del core_copy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    @staticmethod
    def load_model(load_path, device="cuda"):
        """直接加载合并后的模型"""
        return ProteinLanguageModel.load_model_from_checkpoint(load_path, device)


if __name__ == "__main__":
    new_files = ["/home/hubowen/work/protein_data/Dataset83.h5ad", "/home/hubowen/work/protein_data/Dataset101.h5ad"]
    model = ProteinLanguageModel.load_model_from_checkpoint("/home/hubowen/work/protein_llm/checkpoints/basemodel/large-scale-new-9/model_749999.pt")
    finetuner = AddNewProteinFineTuner(
        model=model,  
        adata=new_files,
        device="cuda",
        lr=1e-4,
        batch_size=16,
        masked_ratio=0.15,
        expr_name="finetune_test"
    )
    finetuner.fit(
        num_epochs=5,
        save=True,
        save_dir="./checkpoints_finetune",
        save_epoch_interval=1,
        early_stop=True,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.1
    )
    
    finetuner.plot_loss_history(save_path="./finetune_loss.png")
    test1 = AddNewProteinFineTuner.load_model("/home/hubowen/work/protein_llm/checkpoints_finetune/finetune_test_finetune_final.pt", device="cuda")
    print(test1)