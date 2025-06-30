// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'backtest_summary.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

BacktestSummary _$BacktestSummaryFromJson(Map<String, dynamic> json) =>
    BacktestSummary(
      ticker: json['ticker'] as String,
      returnPct: (json['return_pct'] as num).toDouble(),
      sharpeRatio: (json['sharpe_ratio'] as num).toDouble(),
      maxDrawdown: (json['max_drawdown'] as num).toDouble(),
      winRate: (json['win_rate'] as num).toDouble(),
      profitFactor: (json['profit_factor'] as num).toDouble(),
      totalTrades: (json['total_trades'] as num).toInt(),
      avgIvEntry: (json['avg_iv_entry'] as num).toDouble(),
      premiumCollected: (json['premium_collected'] as num).toDouble(),
    );

Map<String, dynamic> _$BacktestSummaryToJson(BacktestSummary instance) =>
    <String, dynamic>{
      'ticker': instance.ticker,
      'return_pct': instance.returnPct,
      'sharpe_ratio': instance.sharpeRatio,
      'max_drawdown': instance.maxDrawdown,
      'win_rate': instance.winRate,
      'profit_factor': instance.profitFactor,
      'total_trades': instance.totalTrades,
      'avg_iv_entry': instance.avgIvEntry,
      'premium_collected': instance.premiumCollected,
    };
