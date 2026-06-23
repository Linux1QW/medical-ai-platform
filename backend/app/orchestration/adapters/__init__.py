from app.orchestration.adapters.registry import register_adapter
from app.orchestration.adapters.inquiry import InquiryAdapter
from app.orchestration.adapters.diagnosis import DiagnosisAdapter
from app.orchestration.adapters.treatment import TreatmentAdapter
from app.orchestration.adapters.knowledge import KnowledgeAdapter
from app.orchestration.adapters.humanistic import HumanisticAdapter


def register_all():
    """注册所有 Agent 适配器"""
    register_adapter(InquiryAdapter())
    register_adapter(DiagnosisAdapter())
    register_adapter(TreatmentAdapter())
    register_adapter(KnowledgeAdapter())
    register_adapter(HumanisticAdapter())
