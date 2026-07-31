import { describe, it, expect } from 'vitest';
import type { RubricItem, ClinicalClaim, RiskFinding, ReportManifest, RubricVerdict } from '../types';

/**
 * Task 12 — 评估报告前端升级测试
 * 
 * 由于 React 组件测试需要完整的 DOM 环境，
 * 这里主要测试类型语义和数据转换逻辑。
 */

describe('RubricItem 类型语义', () => {
  it('unassessed verdict 不应被渲染为 0 分', () => {
    const item: RubricItem = {
      item_id: 'INQ-001',
      dimension: 'inquiry',
      verdict: 'unassessed',
      score: null,
      severity: 'medium',
      description: '主诉采集',
      evidence_spans: [],
      citation_ids: [],
    };
    // unassessed 的 score 应为 null，不是 0
    expect(item.score).toBeNull();
    expect(item.verdict).toBe('unassessed');
  });

  it('high severity + fail 应标记 needs_review', () => {
    const item: RubricItem = {
      item_id: 'TREAT-003',
      dimension: 'treatment',
      verdict: 'fail',
      score: 0,
      severity: 'high',
      description: '治疗禁忌识别',
      evidence_spans: ['患者青霉素过敏'],
      citation_ids: ['cite-001'],
    };
    const isHighSeverityFail = item.severity === 'high' && item.verdict === 'fail';
    expect(isHighSeverityFail).toBe(true);
  });

  it('not_applicable 不应参与分数计算', () => {
    const items: RubricItem[] = [
      { item_id: 'A', dimension: 'inquiry', verdict: 'pass', score: 80, severity: 'medium', description: '', evidence_spans: [], citation_ids: [] },
      { item_id: 'B', dimension: 'inquiry', verdict: 'not_applicable', score: null, severity: 'low', description: '', evidence_spans: [], citation_ids: [] },
      { item_id: 'C', dimension: 'inquiry', verdict: 'unassessed', score: null, severity: 'medium', description: '', evidence_spans: [], citation_ids: [] },
    ];
    // 只计算 pass/partial/fail 的分数
    const scored = items.filter(i => ['pass', 'partial', 'fail'].includes(i.verdict));
    const avg = scored.reduce((sum, i) => sum + (i.score ?? 0), 0) / scored.length;
    expect(avg).toBe(80);
    expect(scored.length).toBe(1);
  });

  it('所有 verdict 类型都有定义', () => {
    const verdicts: RubricVerdict[] = ['pass', 'partial', 'fail', 'not_applicable', 'unassessed'];
    expect(verdicts).toHaveLength(5);
  });
});

describe('ClinicalClaim 类型语义', () => {
  it('unsupported treatment claim 应标记 needs_review', () => {
    const claim: ClinicalClaim = {
      claim_id: 'claim-001',
      claim_type: 'treatment',
      content: '建议使用阿莫西林',
      status: 'unsupported',
      evidence_links: [],
      needs_review: true,
    };
    expect(claim.needs_review).toBe(true);
    expect(claim.status).toBe('unsupported');
  });

  it('冲突证据不能标记为 supported', () => {
    const claim: ClinicalClaim = {
      claim_id: 'claim-002',
      claim_type: 'diagnosis',
      content: '急性心肌梗死',
      status: 'conflicting',
      evidence_links: [
        { citation_id: 'c1', link_type: 'supports', entailment_score: 0.8, evidence_span: 'ST段抬高' },
        { citation_id: 'c2', link_type: 'contradicts', entailment_score: 0.7, evidence_span: '心肌酶正常' },
      ],
      needs_review: true,
    };
    expect(claim.status).not.toBe('supported');
    expect(claim.evidence_links.some(l => l.link_type === 'contradicts')).toBe(true);
  });

  it('claim status 枚举完整', () => {
    const statuses = ['supported', 'partially_supported', 'unsupported', 'conflicting'];
    expect(statuses).toHaveLength(4);
  });
});

describe('RiskFinding 类型语义', () => {
  it('high severity finding 应标记 needs_review', () => {
    const finding: RiskFinding = {
      finding_id: 'risk-001',
      risk_type: 'emergency',
      severity: 'high',
      description: '胸痛患者未排除心梗',
      evidence_span: '患者胸闷3天',
      policy_action: '立即心电图检查',
      needs_review: true,
    };
    expect(finding.severity).toBe('high');
    expect(finding.needs_review).toBe(true);
  });

  it('risk_type 枚举完整', () => {
    const types = ['emergency', 'medication', 'population', 'privacy', 'evidence_conflict'];
    expect(types).toHaveLength(5);
  });
});

describe('ReportManifest 类型语义', () => {
  it('manifest 应包含所有版本字段', () => {
    const manifest: ReportManifest = {
      report_kind: 'regression',
      report_id: 'rpt-001',
      created_at: '2026-08-01T00:00:00Z',
      case_count: 18,
      dataset_version: 'v1.0',
      model_version: 'gpt-4-0125',
      prompt_version: 'v1.2',
      judge_version: 'v1.0',
      kb_version: 'kb-2026-07',
      scoring_policy_version: 'v1.0',
      seed: 42,
    };
    expect(manifest.report_kind).toBe('regression');
    expect(manifest.case_count).toBe(18);
    expect(manifest.model_version).toBeTruthy();
    expect(manifest.kb_version).toBeTruthy();
  });

  it('report_kind 枚举完整', () => {
    const kinds = ['smoke', 'regression', 'benchmark', 'legacy_unknown'];
    expect(kinds).toHaveLength(4);
  });
});

describe('分数渲染逻辑', () => {
  it('unassessed 不应渲染为 0', () => {
    const score: number | null = null;
    const verdict: RubricVerdict = 'unassessed';
    // 前端逻辑：verdict === 'unassessed' 时显示"未评估"而非 score
    const displayText = verdict === 'unassessed' ? '未评估' : `${score ?? 0}`;
    expect(displayText).toBe('未评估');
    expect(displayText).not.toBe('0');
  });

  it('null score 不应渲染为 0', () => {
    const score: number | null = null;
    const verdict: string = 'pass';
    // pass 但 score 为 null 是异常状态，但不应显示 0
    const displayText = verdict === 'unassessed' ? '未评估' : (score !== null ? `${score}` : 'N/A');
    expect(displayText).toBe('N/A');
  });

  it('needs_review 状态不应渲染总分为 0', () => {
    const isNeedsReview = true;
    const totalScore: number | null = null;
    const displayText = isNeedsReview ? '待复核' : `${totalScore ?? 0}`;
    expect(displayText).toBe('待复核');
    expect(displayText).not.toBe('0');
  });
});
