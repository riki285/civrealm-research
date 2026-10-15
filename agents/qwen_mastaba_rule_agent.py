from .qwen_mastaba_agent import QwenMastabaAgent
from .workers.rule_based_worker import RuleBasedWorker, aggregate_safety_rejections


class QwenMastabaRuleAgent(QwenMastabaAgent):
    """
    advisor は LLM（Qwen）、controller はルールベースで動作するエージェント。
    QwenMastabaAgent の _create_controller_worker() のみをオーバーライドする。
    """

    def _create_controller_worker(self, entity_type, entity_id):
        # agent=self を渡すことで、worker が安全弁の発動をエージェント側に
        # 集計できるようになる（aggregate_safety_rejections で参照する）。
        return RuleBasedWorker(ctrl_type=entity_type,
                               actor_id=entity_id,
                               agent=self)

    def get_safety_stats(self):
        """安全弁（破滅的行動の拒否）の発動状況を返す。main.py の記録用。"""
        return aggregate_safety_rejections(self)