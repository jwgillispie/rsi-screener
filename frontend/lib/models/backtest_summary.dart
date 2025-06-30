import 'package:json_annotation/json_annotation.dart';

part 'backtest_summary.g.dart';

@JsonSerializable()
class BacktestSummary {
  final String ticker;
  @JsonKey(name: 'return_pct')
  final double returnPct;
  @JsonKey(name: 'sharpe_ratio')
  final double sharpeRatio;
  @JsonKey(name: 'max_drawdown')
  final double maxDrawdown;
  @JsonKey(name: 'win_rate')
  final double winRate;
  @JsonKey(name: 'profit_factor')
  final double profitFactor;
  @JsonKey(name: 'total_trades')
  final int totalTrades;
  @JsonKey(name: 'avg_iv_entry')
  final double avgIvEntry;
  @JsonKey(name: 'premium_collected')
  final double premiumCollected;

  BacktestSummary({
    required this.ticker,
    required this.returnPct,
    required this.sharpeRatio,
    required this.maxDrawdown,
    required this.winRate,
    required this.profitFactor,
    required this.totalTrades,
    required this.avgIvEntry,
    required this.premiumCollected,
  });

  factory BacktestSummary.fromJson(Map<String, dynamic> json) =>
      _$BacktestSummaryFromJson(json);

  Map<String, dynamic> toJson() => _$BacktestSummaryToJson(this);

  // Helper methods for UI
  String get performanceGrade {
    if (returnPct >= 30 && sharpeRatio >= 1.0) return 'A+';
    if (returnPct >= 20 && sharpeRatio >= 0.8) return 'A';
    if (returnPct >= 15 && sharpeRatio >= 0.6) return 'B+';
    if (returnPct >= 10 && sharpeRatio >= 0.4) return 'B';
    if (returnPct >= 5) return 'C';
    return 'D';
  }

  String get riskLevel {
    if (maxDrawdown <= 5) return 'Low';
    if (maxDrawdown <= 10) return 'Moderate';
    if (maxDrawdown <= 20) return 'High';
    return 'Very High';
  }

  String get premiumCollectedFormatted {
    if (premiumCollected >= 1000000) {
      return '\$${(premiumCollected / 1000000).toStringAsFixed(1)}M';
    } else if (premiumCollected >= 1000) {
      return '\$${(premiumCollected / 1000).toStringAsFixed(0)}K';
    }
    return '\$${premiumCollected.toStringAsFixed(2)}';
  }

  bool get isPerformer => returnPct > 15 && sharpeRatio > 0.6;
}