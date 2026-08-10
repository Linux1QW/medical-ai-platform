"""Generate 72-case coach benchmark JSONL dataset."""
import json
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "coach_v1.jsonl")

# Ordering scheme to make case_033 = psychiatry/medium/evasive
SPECIALTIES = ["internal", "surgery", "psychiatry", "pediatrics", "obstetrics_gynecology", "emergency"]
DIFFICULTIES = ["hard", "easy", "medium"]  # within each specialty
PERSONALITIES = ["evasive", "cooperative", "anxious", "elderly_confused"]

# Fixed case_033
FIXED_CASE_033 = {
    "case_id": "case_033",
    "specialty": "psychiatry",
    "difficulty": "medium",
    "personality": "evasive",
    "visible_context": {
        "patient_age": 32,
        "patient_gender": "女",
        "chief_complaint": "失眠伴情绪低落三个月"
    },
    "dialogue_prefix": [
        {"role": "doctor", "content": "你好，请坐。请问最近有什么困扰你的事情吗？"},
        {"role": "patient", "content": "嗯……就是睡不好，没什么精神。"},
        {"role": "doctor", "content": "这种情况持续多久了？有没有什么具体的原因？"},
        {"role": "patient", "content": "大概三个月吧，也没什么特别的原因，就是觉得累。"}
    ],
    "expected_intent": "hpi_associated",
    "expected_stage": "history_present_illness",
    "forbidden_hidden_facts": [
        "expected_diagnosis:重度抑郁发作",
        "unrevealed:半年前曾有自杀意念",
        "unrevealed:目前正在服用抗抑郁药物"
    ],
    "target_slots": ["hpi_associated", "hpi_progression", "past_history"]
}

# Clinical case definitions: (age, gender, chief_complaint, dialogue, expected_intent, expected_stage, forbidden, target_slots)
# Dialogue: list of (role, content) tuples, 2-4 turns

CASES_DATA = {
    # ============ INTERNAL (内科) ============
    # hard × 4
    ("internal", "hard", "evasive"): (68, "男", "反复咳嗽半年，加重伴喘息一周", [
        ("doctor", "您好，请坐。看您病历上写咳嗽有半年了？"),
        ("patient", "嗯，咳了挺久了。"),
        ("doctor", "最近一周好像加重了？有没有喘息的感觉？"),
        ("patient", "是有点喘，不过还行，能忍受。")
    ], "hpi_progression", "history_present_illness",
    ["expected_diagnosis:慢性阻塞性肺疾病急性加重", "unrevealed:患者有40年吸烟史但否认", "unrevealed:近一月出现下肢水肿"],
    ["hpi_progression", "hpi_severity", "hpi_timing"]),

    ("internal", "hard", "cooperative"): (72, "女", "头晕伴双下肢水肿两周", [
        ("doctor", "阿姨您好，请问您最近有什么不舒服？"),
        ("patient", "大夫我头晕得厉害，两条腿也肿了。"),
        ("doctor", "头晕是天旋地转的还是昏昏沉沉的？"),
        ("patient", "就是昏昏沉沉的，走路有点飘。腿肿是最近两周才有的。")
    ], "past_history", "history_present_illness",
    ["expected_diagnosis:慢性心力衰竭急性加重", "unrevealed:患者有高血压病史20年未规律服药", "unrevealed:近期自行服用偏方"],
    ["past_history", "medication", "examination"]),

    ("internal", "hard", "anxious"): (55, "男", "胸闷心悸反复发作一月", [
        ("doctor", "您好，请说说您的情况。"),
        ("patient", "大夫我是不是得了心脏病？我胸口老是闷，心跳又快。"),
        ("doctor", "别紧张，胸闷一般在什么时候出现？"),
        ("patient", "有时候累了就有，有时候半夜也会。我真的很担心，会不会是心梗？")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:焦虑症伴躯体化症状", "unrevealed:患者近期工作压力极大", "unrevealed:已在外院做过心脏检查均正常"],
    ["examination", "hpi_associated", "past_history"]),

    ("internal", "hard", "elderly_confused"): (80, "男", "意识模糊伴发热两天", [
        ("doctor", "老人家您好，请问您哪里不舒服？"),
        ("patient", "啊？我……我有点发烧，头也糊里糊涂的。"),
        ("doctor", "您发烧几天了？家里人说您这两天有点迷糊？"),
        ("patient", "几天？我也不太记得了……好像是前天开始的吧。我这是怎么了？")
    ], "hpi_timing", "history_present_illness",
    ["expected_diagnosis:肺部感染伴谵妄", "unrevealed:患者有糖尿病史血糖控制差", "unrevealed:家属反映患者近一周尿量明显减少"],
    ["hpi_timing", "hpi_associated", "examination"]),

    # easy × 4
    ("internal", "easy", "evasive"): (45, "男", "反复上腹部隐痛两周", [
        ("doctor", "您好，请问有什么不舒服？"),
        ("patient", "胃有点不舒服，隐隐地痛。"),
        ("doctor", "疼痛跟吃饭有关系吗？饭前还是饭后？"),
        ("patient", "好像是饭后多一点。")
    ], "hpi_onset", "history_present_illness",
    ["expected_diagnosis:胃溃疡", "unrevealed:患者长期服用NSAIDs止痛药", "unrevealed:有幽门螺杆菌感染史"],
    ["hpi_onset", "hpi_progression", "hpi_severity"]),

    ("internal", "easy", "cooperative"): (55, "男", "反复头痛一周", [
        ("doctor", "您好，请坐。请问您哪里不舒服？"),
        ("patient", "我最近一周总是头痛。"),
        ("doctor", "头痛是从什么时候开始的？"),
        ("patient", "大概一周前开始的，一开始不太严重，后来越来越频繁。")
    ], "hpi_onset", "history_present_illness",
    ["expected_diagnosis:偏头痛", "unrevealed:家族史中有脑瘤", "unrevealed:近期视力出现模糊"],
    ["hpi_onset", "hpi_progression", "hpi_severity"]),

    ("internal", "easy", "anxious"): (38, "女", "心慌手抖伴体重下降一月", [
        ("doctor", "您好，请问您有什么不舒服？"),
        ("patient", "大夫我最近老是心慌，手也抖，体重还掉了好几斤。"),
        ("doctor", "这种情况多久了？"),
        ("patient", "大概一个月了，我是不是得了什么严重的病？")
    ], "hpi_associated", "history_present_illness",
    ["expected_diagnosis:甲状腺功能亢进", "unrevealed:患者家族有甲状腺疾病史", "unrevealed:近期月经量明显减少"],
    ["hpi_associated", "hpi_progression", "past_history"]),

    ("internal", "easy", "elderly_confused"): (75, "女", "血糖升高十年，近期控制不佳", [
        ("doctor", "阿姨您好，今天来复查血糖吗？"),
        ("patient", "是啊大夫，我这血糖老是高，药也吃了。"),
        ("doctor", "您现在在吃什么药？"),
        ("patient", "就是那个二甲双胍，还有……还有一个白色的药，我记不太清了。")
    ], "medication", "history_present_illness",
    ["expected_diagnosis:2型糖尿病血糖控制不佳", "unrevealed:患者经常漏服药物", "unrevealed:近期出现足部麻木"],
    ["medication", "past_history", "hpi_associated"]),

    # medium × 4
    ("internal", "medium", "evasive"): (50, "男", "间断性腹痛伴大便习惯改变两月", [
        ("doctor", "您好，请问您哪里不舒服？"),
        ("patient", "肚子有时候疼，大便也不太正常。"),
        ("doctor", "大便具体怎么不正常？有没有便血？"),
        ("patient", "就……有时候稀一点，别的还好。")
    ], "hpi_progression", "history_present_illness",
    ["expected_diagnosis:结肠癌", "unrevealed:患者有结肠癌家族史但隐瞒", "unrevealed:近两月体重下降8公斤"],
    ["hpi_progression", "hpi_severity", "family_social"]),

    ("internal", "medium", "cooperative"): (60, "女", "反复关节疼痛三年，加重一月", [
        ("doctor", "阿姨您好，关节疼了三年了？"),
        ("patient", "是的，两只手和膝盖都疼，早上起来还僵硬。"),
        ("doctor", "早上僵硬大概持续多久？"),
        ("patient", "差不多一个小时才能活动开。最近一个月疼得更厉害了。")
    ], "medication", "history_present_illness",
    ["expected_diagnosis:类风湿关节炎", "unrevealed:患者曾使用偏方导致肝功能异常", "unrevealed:有结核病史"],
    ["medication", "past_history", "examination"]),

    ("internal", "medium", "anxious"): (42, "男", "体检发现肝功能异常一周", [
        ("doctor", "您好，请坐。看您体检报告转氨酶偏高？"),
        ("patient", "对，大夫，我看了网上说转氨酶高可能是肝炎或者肝癌，我很害怕。"),
        ("doctor", "先别担心，您平时喝酒吗？"),
        ("patient", "应酬比较多，一周大概三四次。这会不会是喝酒喝的？")
    ], "past_history", "history_present_illness",
    ["expected_diagnosis:酒精性肝病", "unrevealed:患者实际每日饮酒量远超所述", "unrevealed:有乙肝病毒携带史未告知"],
    ["past_history", "medication", "hpi_associated"]),

    ("internal", "medium", "elderly_confused"): (78, "男", "记忆力下降一年，加重伴行为异常两月", [
        ("doctor", "老人家您好，家里人说说您最近怎么样？"),
        ("patient", "我挺好的啊，就是家里人老说我忘事。"),
        ("doctor", "您最近有没有忘记关火或者出门找不到路的情况？"),
        ("patient", "好像有过一两次……也不记得了。我觉得我没什么问题。")
    ], "family_social", "history_present_illness",
    ["expected_diagnosis:阿尔茨海默病", "unrevealed:家属反映患者曾走失一次", "unrevealed:患者有脑血管病危险因素"],
    ["family_social", "hpi_progression", "examination"]),

    # ============ SURGERY (外科) ============
    # hard × 4
    ("surgery", "hard", "evasive"): (58, "男", "右下肢疼痛伴间歇性跛行半年", [
        ("doctor", "您好，请说说您的情况。"),
        ("patient", "腿疼，走一段路就得停下来歇歇。"),
        ("doctor", "大概走多远需要停下来？"),
        ("patient", "也就两三百米吧……可能更短。")
    ], "past_history", "history_present_illness",
    ["expected_diagnosis:下肢动脉硬化闭塞症", "unrevealed:患者有长期大量吸烟史", "unrevealed:左下肢也有症状但未提及"],
    ["past_history", "hpi_progression", "examination"]),

    ("surgery", "hard", "cooperative"): (45, "男", "右上腹剧痛伴恶心呕吐六小时", [
        ("doctor", "您好，看您捂着肚子，能说说怎么疼的吗？"),
        ("patient", "右上腹特别疼，一阵一阵的，还吐了好几次。"),
        ("doctor", "疼痛有没有往肩膀或者后背放射？"),
        ("patient", "好像右肩膀也有点酸疼。今天早上吃了油腻的东西后开始的。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:急性胆囊炎伴胆石症", "unrevealed:患者既往有类似发作未就医", "unrevealed:有糖尿病史影响手术决策"],
    ["examination", "past_history", "treatment_communication"]),

    ("surgery", "hard", "anxious"): (35, "女", "发现右侧乳房肿块一周", [
        ("doctor", "您好，请坐。您发现肿块一周了？"),
        ("patient", "是的大夫，我洗澡的时候摸到的，我好害怕是不是癌症。"),
        ("doctor", "肿块大概多大？活动度怎么样？"),
        ("patient", "大概花生米大小，好像能推动。大夫您一定要帮帮我。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:乳腺纤维腺瘤", "unrevealed:患者母亲有乳腺癌病史", "unrevealed:肿块近期增大明显"],
    ["examination", "family_social", "hpi_associated"]),

    ("surgery", "hard", "elderly_confused"): (76, "男", "摔倒后右髋部疼痛不能站立两小时", [
        ("doctor", "老人家您好，您是怎么摔的？"),
        ("patient", "我……上厕所的时候滑了一下，右边屁股这就疼得站不起来了。"),
        ("doctor", "您当时有没有碰到头？有没有头晕？"),
        ("patient", "头好像没有……我也不太记得了。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:股骨颈骨折", "unrevealed:患者有严重骨质疏松", "unrevealed:患者长期服用抗凝药物"],
    ["examination", "past_history", "medication"]),

    # easy × 4
    ("surgery", "easy", "evasive"): (40, "男", "肛门疼痛伴便血三天", [
        ("doctor", "您好，请问您有什么不舒服？"),
        ("patient", "上厕所的时候有点疼，还有血。"),
        ("doctor", "血是什么颜色的？量多吗？"),
        ("patient", "鲜红色的，量不算多，就手纸上有。")
    ], "hpi_associated", "history_present_illness",
    ["expected_diagnosis:肛裂", "unrevealed:患者有长期便秘史", "unrevealed:既往有肛周脓肿手术史"],
    ["hpi_associated", "hpi_progression", "past_history"]),

    ("surgery", "easy", "cooperative"): (30, "男", "转移性右下腹痛12小时", [
        ("doctor", "您好，听说您肚子疼？能详细说说吗？"),
        ("patient", "一开始是肚脐周围疼，后来跑到右下腹了。"),
        ("doctor", "大概什么时候开始转移到右下腹的？"),
        ("patient", "大概今天早上开始的，到现在越来越疼了。")
    ], "hpi_associated", "history_present_illness",
    ["expected_diagnosis:急性阑尾炎", "unrevealed:患者有克罗恩病史", "unrevealed:近期有腹泻伴发热"],
    ["hpi_associated", "examination", "past_history"]),

    ("surgery", "easy", "anxious"): (50, "女", "发现颈部肿块两周", [
        ("doctor", "您好，您发现颈部有肿块？"),
        ("patient", "对，在脖子前面，我担心是不是甲状腺出了问题。"),
        ("doctor", "肿块有没有随着吞咽上下移动？"),
        ("patient", "好像会的。大夫这不会是甲状腺癌吧？")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:甲状腺结节", "unrevealed:患者有甲状腺癌家族史", "unrevealed:近期有声嘶症状"],
    ["examination", "hpi_associated", "family_social"]),

    ("surgery", "easy", "elderly_confused"): (70, "男", "排尿困难伴尿频尿急半年", [
        ("doctor", "老人家您好，听说您排尿有困难？"),
        ("patient", "是啊，尿尿很费劲，而且老是要跑厕所。"),
        ("doctor", "晚上起来几次？"),
        ("patient", "三四次吧，睡不好觉。")
    ], "hpi_progression", "history_present_illness",
    ["expected_diagnosis:良性前列腺增生", "unrevealed:患者有急性尿潴留发作史", "unrevealed:PSA指标轻度升高"],
    ["hpi_progression", "hpi_severity", "examination"]),

    # medium × 4
    ("surgery", "medium", "evasive"): (55, "男", "左腰部酸痛伴肉眼血尿两天", [
        ("doctor", "您好，请问您哪里不舒服？"),
        ("patient", "腰有点酸疼，小便颜色也不太对。"),
        ("doctor", "小便是什么颜色？有没有血块？"),
        ("patient", "偏红吧，血块倒是没有。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:输尿管结石", "unrevealed:患者有痛风病史", "unrevealed:既往有类似发作自行缓解"],
    ["examination", "hpi_associated", "past_history"]),

    ("surgery", "medium", "cooperative"): (28, "男", "右膝扭伤后肿痛两小时", [
        ("doctor", "您好，膝盖是怎么伤的？"),
        ("patient", "打篮球的时候扭了一下，马上就肿起来了。"),
        ("doctor", "当时有没有听到响声？"),
        ("patient", "好像有'咯噔'一声，然后膝盖就软了，站不住。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:前交叉韧带损伤", "unrevealed:患者既往有同侧膝关节损伤史", "unrevealed:膝关节存在明显不稳感"],
    ["examination", "hpi_associated", "treatment_communication"]),

    ("surgery", "medium", "anxious"): (48, "女", "发现腹股沟区可复性包块一月", [
        ("doctor", "您好，您说发现了一个包块？"),
        ("patient", "对，在右边大腿根，站着就有，躺下就没了。"),
        ("doctor", "包块大不大？有没有疼痛？"),
        ("patient", "鸡蛋大小吧，不太疼。但我上网查了，很担心是不是肿瘤。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:腹股沟疝", "unrevealed:患者有慢性咳嗽史", "unrevealed:包块曾出现嵌顿后自行回纳"],
    ["examination", "hpi_associated", "treatment_communication"]),

    ("surgery", "medium", "elderly_confused"): (82, "女", "左肩部摔伤后疼痛伴活动受限一天", [
        ("doctor", "阿姨您好，您肩膀是怎么伤的？"),
        ("patient", "昨天走路不小心摔了一跤，左手撑着地，然后肩膀就疼得不行。"),
        ("doctor", "现在胳膊能抬起来吗？"),
        ("patient", "抬不起来，一动就疼。我年纪大了，是不是骨头断了？")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:肱骨近端骨折", "unrevealed:患者有严重骨质疏松长期未治疗", "unrevealed:患者有房颤服用抗凝药"],
    ["examination", "past_history", "medication"]),

    # ============ PSYCHIATRY (精神科) ============
    # hard × 4
    ("psychiatry", "hard", "evasive"): (28, "男", "行为异常伴言语增多一月", [
        ("doctor", "你好，请坐。能跟我说说最近怎么样吗？"),
        ("patient", "我挺好的啊，就是精力比较充沛，想法比较多。"),
        ("doctor", "家里人怎么说您的状态？"),
        ("patient", "他们就是大惊小怪，我觉得我现在状态特别好。")
    ], "hpi_associated", "history_present_illness",
    ["expected_diagnosis:双相情感障碍躁狂发作", "unrevealed:患者半月来几乎未睡眠", "unrevealed:有大量不理性消费行为"],
    ["hpi_associated", "hpi_progression", "family_social"]),

    ("psychiatry", "hard", "cooperative"): (45, "女", "反复出现消极想法半年", [
        ("doctor", "您好，请坐。能跟我说说您的感受吗？"),
        ("patient", "大夫，我总觉得活着没意思，老是想到死。"),
        ("doctor", "您有没有具体的计划？"),
        ("patient", "有时候会想，但没有真的去做。我愿意配合治疗。")
    ], "hpi_severity", "history_present_illness",
    ["expected_diagnosis:重度抑郁障碍", "unrevealed:患者已写好遗书", "unrevealed:有酒精依赖问题"],
    ["hpi_severity", "past_history", "medication"]),

    ("psychiatry", "hard", "anxious"): (22, "女", "反复检查门锁是否关好两年，加重半年", [
        ("doctor", "你好，能跟我说说你反复检查的情况吗？"),
        ("patient", "我每次出门都要检查门锁好几遍，有时候要检查十几分钟。"),
        ("doctor", "如果不检查会怎么样？"),
        ("patient", "会非常焦虑，觉得家里会被盗，控制不住。我是不是疯了？")
    ], "hpi_progression", "history_present_illness",
    ["expected_diagnosis:强迫症", "unrevealed:患者检查行为已严重影响工作", "unrevealed:有完美主义人格特质"],
    ["hpi_progression", "hpi_severity", "family_social"]),

    ("psychiatry", "hard", "elderly_confused"): (65, "男", "疑心重伴睡眠障碍一年", [
        ("doctor", "老人家您好，家里人带您来看诊，能说说怎么回事吗？"),
        ("patient", "我觉得有人在监视我，邻居好像在密谋什么。"),
        ("doctor", "您能具体说说吗？有什么证据？"),
        ("patient", "我……我也说不上来，就是感觉。晚上也睡不好，老是听到有人说话。")
    ], "hpi_associated", "history_present_illness",
    ["expected_diagnosis:晚发性精神分裂症", "unrevealed:患者有听幻觉和被害妄想", "unrevealed:有脑血管病危险因素"],
    ["hpi_associated", "examination", "past_history"]),

    # easy × 4
    ("psychiatry", "easy", "evasive"): (35, "男", "情绪波动伴社交退缩半年", [
        ("doctor", "你好，请坐。最近感觉怎么样？"),
        ("patient", "还行吧，就是不太想出门。"),
        ("doctor", "不想出门的原因是什么？"),
        ("patient", "就是觉得没意思，不想见人。")
    ], "hpi_progression", "history_present_illness",
    ["expected_diagnosis:社交焦虑障碍", "unrevealed:患者有惊恐发作经历", "unrevealed:正在使用酒精缓解焦虑"],
    ["hpi_progression", "hpi_associated", "family_social"]),

    ("psychiatry", "easy", "cooperative"): (30, "女", "情绪低落伴兴趣减退两月", [
        ("doctor", "您好，请坐。能跟我说说最近的心情吗？"),
        ("patient", "最近两个月总是高兴不起来，以前喜欢的事情也不想做了。"),
        ("doctor", "食欲和睡眠怎么样？"),
        ("patient", "食欲不太好，睡得也少，凌晨三四点就醒了。")
    ], "hpi_associated", "history_present_illness",
    ["expected_diagnosis:中度抑郁发作", "unrevealed:患者近期经历了离婚", "unrevealed:有自伤行为"],
    ["hpi_associated", "hpi_timing", "past_history"]),

    ("psychiatry", "easy", "anxious"): (40, "男", "紧张不安伴心悸出汗三月", [
        ("doctor", "您好，请问您紧张不安的情况能具体说说吗？"),
        ("patient", "我总是无缘无故地紧张，心跳很快，还出冷汗。"),
        ("doctor", "这种紧张有没有特定的触发场景？"),
        ("patient", "好像没有特定的，有时候在办公室就突然开始了。")
    ], "hpi_associated", "history_present_illness",
    ["expected_diagnosis:广泛性焦虑障碍", "unrevealed:患者有惊恐发作史", "unrevealed:家族有焦虑症病史"],
    ["hpi_associated", "hpi_progression", "past_history"]),

    ("psychiatry", "easy", "elderly_confused"): (70, "女", "情绪低落伴记忆力下降半年", [
        ("doctor", "阿姨您好，您觉得自己最近怎么样？"),
        ("patient", "不太好，老是忘事，心情也不好。"),
        ("doctor", "心情不好具体是什么感觉？"),
        ("patient", "就是高兴不起来，什么都不想干，觉得自己没用了。")
    ], "hpi_associated", "history_present_illness",
    ["expected_diagnosis:老年期抑郁", "unrevealed:患者有脑血管病史", "unrevealed:认知功能下降可能为假性痴呆"],
    ["hpi_associated", "examination", "past_history"]),

    # medium × 4 (case_033 is first)
    ("psychiatry", "medium", "cooperative"): (50, "女", "反复躯体不适检查无异常一年", [
        ("doctor", "您好，请问您有什么不舒服？"),
        ("patient", "我身上到处都不舒服，头疼、胃疼、背也疼。"),
        ("doctor", "做过哪些检查？"),
        ("patient", "该做的都做了，胃镜、CT都做了，医生说没问题。但我确实不舒服。")
    ], "hpi_associated", "history_present_illness",
    ["expected_diagnosis:躯体症状障碍", "unrevealed:患者近期家庭关系紧张", "unrevealed:有焦虑抑郁共病"],
    ["hpi_associated", "past_history", "family_social"]),

    ("psychiatry", "medium", "anxious"): (25, "男", "害怕特定社交场合两年", [
        ("doctor", "你好，能说说你害怕的社交场合是什么样的？"),
        ("patient", "就是人多的地方，比如开会、聚餐，我会特别紧张。"),
        ("doctor", "紧张的时候身体有什么反应？"),
        ("patient", "脸红、手抖、说话结巴，恨不得马上离开。")
    ], "hpi_progression", "history_present_illness",
    ["expected_diagnosis:社交焦虑障碍", "unrevealed:患者因此已辞职半年", "unrevealed:有回避型人格特征"],
    ["hpi_progression", "hpi_severity", "family_social"]),

    ("psychiatry", "medium", "elderly_confused"): (68, "男", "性格改变伴计算力下降两年", [
        ("doctor", "老人家您好，家里人说说您的情况？"),
        ("patient", "我没什么问题啊，就是算账算不太清了。"),
        ("doctor", "家里人觉得您性格有变化吗？"),
        ("patient", "他们说我不像以前了，但我自己觉得挺好的。")
    ], "family_social", "history_present_illness",
    ["expected_diagnosis:额颞叶痴呆", "unrevealed:患者出现不当社交行为", "unrevealed:有额叶影像学改变"],
    ["family_social", "examination", "hpi_progression"]),

    # ============ PEDIATRICS (儿科) ============
    # hard × 4
    ("pediatrics", "hard", "evasive"): (8, "女", "反复腹痛一月，加重三天", [
        ("doctor", "小朋友你好，肚子哪里疼呀？"),
        ("patient", "就……肚子疼。"),
        ("doctor", "是肚脐周围还是别的地方？"),
        ("patient", "嗯……到处都是。")
    ], "hpi_progression", "history_present_illness",
    ["expected_diagnosis:过敏性紫癜（腹型）", "unrevealed:患儿下肢有散在出血点", "unrevealed:近期有上呼吸道感染史"],
    ["hpi_progression", "examination", "hpi_associated"]),

    ("pediatrics", "hard", "cooperative"): (5, "男", "反复喘息发作三年，加重两天", [
        ("doctor", "小朋友你好，你是喘不上气吗？"),
        ("patient", "嗯，我呼吸有点费劲。"),
        ("doctor", "以前有没有类似的情况？"),
        ("patient", "有的，以前也喘过，妈妈说我小时候湿疹也很厉害。")
    ], "past_history", "history_present_illness",
    ["expected_diagnosis:支气管哮喘急性发作", "unrevealed:患儿父亲有哮喘史", "unrevealed:近期未规律使用吸入药物"],
    ["past_history", "medication", "family_social"]),

    ("pediatrics", "hard", "anxious"): (3, "女", "高热三天伴皮疹一天", [
        ("doctor", "小朋友发烧三天了？身上这些红点是什么时候出来的？"),
        ("patient", "昨天开始的，先发烧然后身上就红了。"),
        ("doctor", "有没有痒的感觉？孩子一直在抓吗？"),
        ("patient", "好像是有点痒，她一直在挠。烧到39度多了。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:幼儿急疹", "unrevealed:患儿未完成计划免疫接种", "unrevealed:有热性惊厥史"],
    ["examination", "hpi_associated", "past_history"]),

    ("pediatrics", "hard", "elderly_confused"): (2, "男", "腹泻呕吐一天伴精神萎靡", [
        ("doctor", "宝宝今天怎么了？"),
        ("patient", "拉肚子，吐了好几次，精神不太好。"),
        ("doctor", "大便什么性状？一天几次？"),
        ("patient", "水样的，今天拉了五六次了。尿也少了。")
    ], "hpi_severity", "history_present_illness",
    ["expected_diagnosis:急性感染性腹泻伴脱水", "unrevealed:患儿有先天性心脏病", "unrevealed:已出现中度脱水体征"],
    ["hpi_severity", "examination", "hpi_timing"]),

    # easy × 4
    ("pediatrics", "easy", "evasive"): (10, "男", "反复鼻塞流涕三月", [
        ("doctor", "小朋友你好，鼻子堵了多久了？"),
        ("patient", "好几个月了，一直流鼻涕。"),
        ("doctor", "鼻涕是什么颜色的？"),
        ("patient", "有时候黄有时候白。")
    ], "hpi_associated", "history_present_illness",
    ["expected_diagnosis:过敏性鼻炎", "unrevealed:患儿有哮喘家族史", "unrevealed:症状有季节性加重规律"],
    ["hpi_associated", "hpi_timing", "family_social"]),

    ("pediatrics", "easy", "cooperative"): (6, "女", "发热咳嗽五天", [
        ("doctor", "小朋友你好，发烧几天了？"),
        ("patient", "五天了，还有点咳嗽。"),
        ("doctor", "咳嗽有痰吗？"),
        ("patient", "有的，黄色的痰。")
    ], "hpi_progression", "history_present_illness",
    ["expected_diagnosis:支气管肺炎", "unrevealed:患儿有免疫缺陷病史", "unrevealed:班级有多名同学类似症状"],
    ["hpi_progression", "examination", "hpi_associated"]),

    ("pediatrics", "easy", "anxious"): (4, "男", "全身反复出现风团伴瘙痒一周", [
        ("doctor", "小朋友身上这些疙瘩是什么时候起的？"),
        ("patient", "一个星期了，起了又消，消了又起。"),
        ("doctor", "有没有吃什么特别的东西？"),
        ("patient", "好像吃了虾以后就开始了。妈妈很担心。")
    ], "hpi_onset", "history_present_illness",
    ["expected_diagnosis:急性荨麻疹", "unrevealed:患儿有特应性体质", "unrevealed:伴有血管性水肿"],
    ["hpi_onset", "hpi_associated", "past_history"]),

    ("pediatrics", "easy", "elderly_confused"): (1, "男", "皮肤黄染三天", [
        ("doctor", "宝宝皮肤发黄三天了？是足月出生的吗？"),
        ("patient", "是的，38周生的。出生时还好，第三天开始黄的。"),
        ("doctor", "吃奶怎么样？"),
        ("patient", "吃奶还行，就是有点嗜睡。")
    ], "hpi_progression", "history_present_illness",
    ["expected_diagnosis:新生儿病理性黄疸", "unrevealed:母婴血型不合", "unrevealed:胆红素水平已达光疗指征"],
    ["hpi_progression", "examination", "hpi_timing"]),

    # medium × 4
    ("pediatrics", "medium", "evasive"): (7, "男", "跛行伴左膝疼痛一周", [
        ("doctor", "小朋友你怎么走路一瘸一拐的？"),
        ("patient", "膝盖疼，不想走路。"),
        ("doctor", "有没有摔过跤？"),
        ("patient", "好像没有……我也不记得了。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:暂时性滑膜炎", "unrevealed:两周前有上呼吸道感染", "unrevealed:需排除化脓性关节炎"],
    ["examination", "hpi_timing", "past_history"]),

    ("pediatrics", "medium", "cooperative"): (12, "女", "身高明显低于同龄人", [
        ("doctor", "你好，爸爸妈妈说你长得比较慢？"),
        ("patient", "嗯，我在班里是最矮的。"),
        ("doctor", "最近一年长了多少？"),
        ("patient", "好像只长了三四厘米。妈妈说她小时候也长得慢。")
    ], "family_social", "history_present_illness",
    ["expected_diagnosis:生长激素缺乏症", "unrevealed:患儿出生时有窒息史", "unrevealed:骨龄明显落后"],
    ["family_social", "examination", "past_history"]),

    ("pediatrics", "medium", "anxious"): (9, "男", "反复头痛伴呕吐两月", [
        ("doctor", "小朋友你头痛多久了？"),
        ("patient", "两个月了，有时候还吐。"),
        ("doctor", "头痛一般在什么时候？"),
        ("patient", "早上起来比较多，吐完会好一点。妈妈很担心。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:颅内占位性病变", "unrevealed:晨起头痛提示颅内压增高", "unrevealed:近期出现视乳头水肿"],
    ["examination", "hpi_timing", "hpi_associated"]),

    ("pediatrics", "medium", "elderly_confused"): (3, "女", "发育倒退伴刻板行为一年", [
        ("doctor", "宝宝以前会说话吗？"),
        ("patient", "以前会说几个词，现在不太说了。"),
        ("doctor", "她平时跟小朋友玩吗？"),
        ("patient", "不太跟别人玩，老是自己转圈圈，拍手。")
    ], "hpi_progression", "history_present_illness",
    ["expected_diagnosis:儿童孤独症谱系障碍", "unrevealed:患儿曾有正常语言发育后倒退", "unrevealed:有感觉统合异常表现"],
    ["hpi_progression", "examination", "family_social"]),

    # ============ OBSTETRICS_GYNECOLOGY (妇产科) ============
    # hard × 4
    ("obstetrics_gynecology", "hard", "evasive"): (30, "女", "停经45天伴少量阴道流血", [
        ("doctor", "您好，停经45天了？有没有做过验孕？"),
        ("patient", "验了，是阳性。"),
        ("doctor", "现在有少量出血，有没有腹痛？"),
        ("patient", "有点隐隐的痛。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:异位妊娠", "unrevealed:患者有盆腔炎病史", "unrevealed:阴道后穹隆穿刺可抽出不凝血"],
    ["examination", "past_history", "hpi_severity"]),

    ("obstetrics_gynecology", "hard", "cooperative"): (35, "女", "孕32周，血压升高伴头痛眼花两天", [
        ("doctor", "您好，您现在怀孕多少周了？"),
        ("patient", "32周了。最近两天血压高，头也疼，看东西有点模糊。"),
        ("doctor", "最近产检血压是多少？"),
        ("patient", "最高到160/110。手脚也有点肿。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:重度子痫前期", "unrevealed:尿蛋白3+", "unrevealed:有发生子痫抽搐的风险"],
    ["examination", "hpi_associated", "treatment_communication"]),

    ("obstetrics_gynecology", "hard", "anxious"): (28, "女", "孕8周发现宫腔内异常回声", [
        ("doctor", "您好，B超发现了什么异常？"),
        ("patient", "医生说宫腔里有个东西，我好怕是宫外孕或者要流产。"),
        ("doctor", "您现在有没有腹痛或出血？"),
        ("patient", "有一点点褐色分泌物，我好担心。")
    ], "assessment_communication", "history_present_illness",
    ["expected_diagnosis:葡萄胎", "unrevealed:血HCG异常升高", "unrevealed:B超呈落雪状图像"],
    ["assessment_communication", "examination", "treatment_communication"]),

    ("obstetrics_gynecology", "hard", "elderly_confused"): (62, "女", "绝经后阴道流血一月", [
        ("doctor", "阿姨您好，您绝经多少年了？"),
        ("patient", "大概十几年了吧。最近一个月下面又出血了。"),
        ("doctor", "出血量大吗？什么颜色？"),
        ("patient", "量不多，暗红色的，有时候是粉红色的。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:子宫内膜癌", "unrevealed:患者有长期雌激素替代治疗史", "unrevealed:B超提示子宫内膜增厚"],
    ["examination", "past_history", "hpi_progression"]),

    # easy × 4
    ("obstetrics_gynecology", "easy", "evasive"): (25, "女", "白带增多伴外阴瘙痒一周", [
        ("doctor", "您好，请问有什么不舒服？"),
        ("patient", "白带比较多，下面有点痒。"),
        ("doctor", "白带是什么颜色、什么性状？"),
        ("patient", "偏黄，有点像豆腐渣。")
    ], "hpi_associated", "history_present_illness",
    ["expected_diagnosis:外阴阴道假丝酵母菌病", "unrevealed:患者近期使用过抗生素", "unrevealed:有糖尿病史"],
    ["hpi_associated", "past_history", "medication"]),

    ("obstetrics_gynecology", "easy", "cooperative"): (28, "女", "停经6周要求产检", [
        ("doctor", "您好，恭喜怀孕。末次月经是什么时候？"),
        ("patient", "大概六周前。"),
        ("doctor", "有没有恶心呕吐等早孕反应？"),
        ("patient", "有的，早上比较明显，其他还好。")
    ], "hpi_associated", "history_present_illness",
    ["expected_diagnosis:早期妊娠", "unrevealed:患者有甲状腺疾病史", "unrevealed:有不良孕产史"],
    ["hpi_associated", "past_history", "examination"]),

    ("obstetrics_gynecology", "easy", "anxious"): (32, "女", "月经不规律伴多毛两年", [
        ("doctor", "您好，月经不规律多久了？"),
        ("patient", "两年了，老是推迟，有时候两三个月才来一次。"),
        ("doctor", "体重有没有变化？"),
        ("patient", "胖了不少，而且脸上老长痘，体毛也多了。我很焦虑。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:多囊卵巢综合征", "unrevealed:患者有胰岛素抵抗", "unrevealed:有不孕诉求"],
    ["examination", "hpi_associated", "family_social"]),

    ("obstetrics_gynecology", "easy", "elderly_confused"): (58, "女", "下腹坠胀伴腰酸三月", [
        ("doctor", "阿姨您好，您觉得哪里不舒服？"),
        ("patient", "肚子下面老是坠坠的，腰也酸。"),
        ("doctor", "有没有感觉下面有东西掉出来？"),
        ("patient", "好像是的，蹲久了更明显。")
    ], "hpi_progression", "history_present_illness",
    ["expected_diagnosis:子宫脱垂", "unrevealed:患者有多次阴道分娩史", "unrevealed:伴有压力性尿失禁"],
    ["hpi_progression", "past_history", "examination"]),

    # medium × 4
    ("obstetrics_gynecology", "medium", "evasive"): (22, "女", "停经50天要求终止妊娠", [
        ("doctor", "您好，您确认怀孕了吗？"),
        ("patient", "验了，是的。我想做人流。"),
        ("doctor", "您之前有没有做过流产？"),
        ("patient", "没有。")
    ], "past_history", "history_present_illness",
    ["expected_diagnosis:早期妊娠要求人工流产", "unrevealed:患者有凝血功能障碍", "unrevealed:既往有药物流产不全史"],
    ["past_history", "examination", "treatment_communication"]),

    ("obstetrics_gynecology", "medium", "cooperative"): (40, "女", "月经量增多伴经期延长半年", [
        ("doctor", "您好，月经量增多多久了？"),
        ("patient", "半年了，以前三天就干净了，现在要七八天，量也多了。"),
        ("doctor", "有没有头晕乏力的感觉？"),
        ("patient", "有的，有时候站起来会头晕。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:子宫肌瘤", "unrevealed:B超提示肌壁间多发肌瘤", "unrevealed:血红蛋白偏低"],
    ["examination", "hpi_associated", "treatment_communication"]),

    ("obstetrics_gynecology", "medium", "anxious"): (45, "女", "体检发现宫颈细胞学异常", [
        ("doctor", "您好，体检报告说宫颈细胞学有异常？"),
        ("patient", "是的，医生说要进一步检查。我是不是得了宫颈癌？"),
        ("doctor", "先别紧张，需要做HPV检测和阴道镜进一步评估。"),
        ("patient", "好的……我真的很害怕。")
    ], "assessment_communication", "history_present_illness",
    ["expected_diagnosis:宫颈上皮内瘤变", "unrevealed:高危型HPV16阳性", "unrevealed:患者有免疫抑制状态"],
    ["assessment_communication", "past_history", "treatment_communication"]),

    ("obstetrics_gynecology", "medium", "elderly_confused"): (55, "女", "下腹隐痛伴腹胀两月", [
        ("doctor", "阿姨您好，您肚子不舒服多久了？"),
        ("patient", "两个月了，肚子胀胀的，有时候有点疼。"),
        ("doctor", "吃饭怎么样？有没有消瘦？"),
        ("patient", "吃不太下，感觉瘦了一点。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:卵巢肿瘤", "unrevealed:CA125明显升高", "unrevealed:有腹水"],
    ["examination", "hpi_associated", "treatment_communication"]),

    # ============ EMERGENCY (急诊) ============
    # hard × 4
    ("emergency", "hard", "evasive"): (40, "男", "意识不清被送急诊", [
        ("doctor", "患者怎么送来的？"),
        ("patient", "我……我没事，就是有点晕。"),
        ("doctor", "身上有酒味，喝了多少酒？"),
        ("patient", "没多少……一点而已。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:急性酒精中毒", "unrevealed:患者同时服用了安眠药", "unrevealed:有头部外伤痕迹"],
    ["examination", "hpi_timing", "past_history"]),

    ("emergency", "hard", "cooperative"): (55, "男", "突发胸痛两小时", [
        ("doctor", "您好，胸痛是什么时候开始的？"),
        ("patient", "两个小时前突然疼的，像压了块石头。"),
        ("doctor", "疼痛有没有往左肩或者下巴放射？"),
        ("patient", "左肩膀确实有点酸，还出了一身冷汗。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:急性ST段抬高型心肌梗死", "unrevealed:患者有冠心病史未规律治疗", "unrevealed:发病前有大量饮酒"],
    ["examination", "past_history", "treatment_communication"]),

    ("emergency", "hard", "anxious"): (30, "女", "误服过量药物两小时", [
        ("doctor", "您好，听说您吃了过量的药？能告诉我吃了什么吗？"),
        ("patient", "我吃了太多的安眠药……我好后悔。"),
        ("doctor", "大概吃了多少？什么时候吃的？"),
        ("patient", "大概半小时前，吃了很多……我不记得具体多少了。")
    ], "hpi_timing", "history_present_illness",
    ["expected_diagnosis:急性药物中毒", "unrevealed:患者有自杀意图", "unrevealed:同时服用了多种药物"],
    ["hpi_timing", "past_history", "assessment_communication"]),

    ("emergency", "hard", "elderly_confused"): (75, "女", "跌倒后意识模糊一小时", [
        ("doctor", "老人家您好，您知道自己在哪吗？"),
        ("patient", "我……我在哪？头好晕。"),
        ("doctor", "您摔了一跤，现在在医院。有没有哪里特别疼？"),
        ("patient", "头……头有点疼，记不太清怎么摔的了。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:急性硬膜下血肿", "unrevealed:患者长期服用华法林", "unrevealed:CT显示中线结构偏移"],
    ["examination", "past_history", "medication"]),

    # easy × 4
    ("emergency", "easy", "evasive"): (25, "男", "右手割伤后出血一小时", [
        ("doctor", "您好，手是怎么伤的？"),
        ("patient", "切菜的时候切到了。"),
        ("doctor", "伤口深不深？有没有伤到肌腱？"),
        ("patient", "应该不深，就是血流得比较多。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:手外伤", "unrevealed:伤口可能污染严重", "unrevealed:患者破伤风疫苗接种史不明"],
    ["examination", "past_history", "treatment_communication"]),

    ("emergency", "easy", "cooperative"): (45, "女", "进食后突发上腹剧痛四小时", [
        ("doctor", "您好，腹痛多久了？"),
        ("patient", "四个小时了，中午吃完饭突然疼起来的。"),
        ("doctor", "疼痛是什么性质的？"),
        ("patient", "像刀割一样，一直疼，弯腰能好一点。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:急性胰腺炎", "unrevealed:患者有胆石症病史", "unrevealed:发病前有暴饮暴食"],
    ["examination", "past_history", "hpi_associated"]),

    ("emergency", "easy", "anxious"): (35, "男", "右小腿动物咬伤后出血两小时", [
        ("doctor", "您好，腿是怎么伤的？"),
        ("patient", "被小区里的狗咬了，突然冲出来咬了一口。"),
        ("doctor", "伤口有没有出血？那只狗有没有打过疫苗？"),
        ("patient", "出血了，不知道狗打没打疫苗。我会不会得什么病？")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:犬咬伤", "unrevealed:伤口较深可能需缝合", "unrevealed:患者有青霉素过敏史影响预防用药"],
    ["examination", "past_history", "treatment_communication"]),

    ("emergency", "easy", "elderly_confused"): (70, "男", "突发左侧肢体无力两小时", [
        ("doctor", "老人家您好，您左手能抬起来吗？"),
        ("patient", "抬不太动……左边胳膊和腿都没什么力气。"),
        ("doctor", "什么时候开始的？"),
        ("patient", "大概两个小时前，突然就不行了。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:急性缺血性脑卒中", "unrevealed:发病时间在溶栓时间窗内", "unrevealed:患者有房颤病史"],
    ["examination", "past_history", "hpi_timing"]),

    # medium × 4
    ("emergency", "medium", "evasive"): (28, "女", "下腹剧痛伴恶心三小时", [
        ("doctor", "您好，肚子怎么疼的？"),
        ("patient", "突然就疼了，在下面。"),
        ("doctor", "有没有可能怀孕？"),
        ("patient", "这个……不太确定。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:卵巢囊肿蒂扭转", "unrevealed:患者已知有卵巢囊肿", "unrevealed:体位改变后突发疼痛"],
    ["examination", "past_history", "hpi_associated"]),

    ("emergency", "medium", "cooperative"): (50, "男", "高温环境下意识模糊四小时", [
        ("doctor", "您好，您是在高温环境下工作的？"),
        ("patient", "对，在工地上干活，今天特别热，后来就头晕得不行。"),
        ("doctor", "有没有恶心呕吐？"),
        ("patient", "吐了一次，现在浑身没力气。")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:热射病", "unrevealed:核心体温超过40°C", "unrevealed:有横纹肌溶解风险"],
    ["examination", "hpi_associated", "treatment_communication"]),

    ("emergency", "medium", "anxious"): (20, "女", "踝关节扭伤后肿痛一小时", [
        ("doctor", "您好，脚踝怎么伤的？"),
        ("patient", "下楼梯的时候崴了一下，马上就肿了。"),
        ("doctor", "能站起来走几步吗？"),
        ("patient", "走不了，太疼了。会不会骨折了？")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:踝关节扭伤", "unrevealed:Ottawa踝关节规则提示需X线", "unrevealed:有韧带完全撕裂可能"],
    ["examination", "hpi_associated", "treatment_communication"]),

    ("emergency", "medium", "elderly_confused"): (80, "男", "发热伴尿频尿急尿痛三天", [
        ("doctor", "老人家您好，发烧几天了？"),
        ("patient", "三天了，尿尿也不舒服，老是想去。"),
        ("doctor", "有没有腰痛？"),
        ("patient", "腰也有点酸疼。我是不是得了什么大病？")
    ], "examination", "history_present_illness",
    ["expected_diagnosis:急性肾盂肾炎", "unrevealed:患者有前列腺增生导致尿潴留", "unrevealed:血象明显升高提示全身感染"],
    ["examination", "past_history", "hpi_associated"]),
}


def build_cases():
    """Build all 72 cases in order."""
    cases = []
    case_num = 1

    for specialty in SPECIALTIES:
        for difficulty in DIFFICULTIES:
            for personality in PERSONALITIES:
                case_id = f"case_{case_num:03d}"

                # Use fixed case_033
                if case_num == 33:
                    cases.append(FIXED_CASE_033)
                    case_num += 1
                    continue

                key = (specialty, difficulty, personality)
                data = CASES_DATA[key]
                age, gender, chief, dialogue_tuples, intent, stage, forbidden, targets = data

                dialogue_prefix = [
                    {"role": role, "content": content}
                    for role, content in dialogue_tuples
                ]

                cases.append({
                    "case_id": case_id,
                    "specialty": specialty,
                    "difficulty": difficulty,
                    "personality": personality,
                    "visible_context": {
                        "patient_age": age,
                        "patient_gender": gender,
                        "chief_complaint": chief,
                    },
                    "dialogue_prefix": dialogue_prefix,
                    "expected_intent": intent,
                    "expected_stage": stage,
                    "forbidden_hidden_facts": forbidden,
                    "target_slots": targets,
                })
                case_num += 1

    return cases


def main():
    cases = build_cases()
    assert len(cases) == 72, f"Expected 72 cases, got {len(cases)}"

    # Verify case_033
    c33 = cases[32]
    assert c33["case_id"] == "case_033"
    assert c33["specialty"] == "psychiatry"
    assert c33["difficulty"] == "medium"
    assert c33["personality"] == "evasive"

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Generated {len(cases)} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
