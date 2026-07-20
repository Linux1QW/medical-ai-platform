"""
敏感数据脱敏工具函数

支持姓名、手机号、身份证等常见字段的脱敏处理。
"""


def mask_name(name: str) -> str:
    """姓名脱敏：张三 → 张*，王小明 → 王*明

    规则：
    - 单字姓名：保持不变
    - 两字姓名：保留姓，名用 * 替代
    - 三字及以上：保留首尾，中间用 * 替代（多个字符用等长 * 替换）
    """
    if not name:
        return name
    if len(name) == 1:
        return name
    if len(name) == 2:
        return name[0] + "*"
    return name[0] + "*" * (len(name) - 2) + name[-1]


def mask_phone(phone: str) -> str:
    """手机号脱敏：13812345678 → 138****5678

    规则：保留前 3 位和后 4 位，中间用 * 替代。
    """
    if not phone:
        return phone
    phone = phone.strip()
    if len(phone) < 7:
        return phone
    return phone[:3] + "*" * (len(phone) - 7) + phone[-4:]


def mask_id_card(id_card: str) -> str:
    """身份证脱敏：110101199001011234 → 110***********1234

    规则：保留前 3 位和后 4 位，中间用 * 替代。
    """
    if not id_card:
        return id_card
    id_card = id_card.strip()
    if len(id_card) < 7:
        return id_card
    return id_card[:3] + "*" * (len(id_card) - 7) + id_card[-4:]


def mask_field(value: str, mask_type: str) -> str:
    """通用脱敏函数

    Args:
        value: 原始值
        mask_type: 脱敏类型，支持 "name" | "phone" | "id_card"

    Returns:
        脱敏后的字符串
    """
    dispatch = {
        "name": mask_name,
        "phone": mask_phone,
        "id_card": mask_id_card,
    }
    masker = dispatch.get(mask_type)
    if masker is None:
        raise ValueError(f"不支持的脱敏类型: {mask_type}，支持: {list(dispatch.keys())}")
    return masker(value)
