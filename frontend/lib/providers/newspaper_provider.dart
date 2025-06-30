import 'package:flutter/foundation.dart';
import '../models/daily_newspaper.dart';
import '../models/backtest_summary.dart';
import '../services/api_service.dart';

class NewspaperProvider with ChangeNotifier {
  DailyNewspaper? _newspaper;
  List<BacktestSummary> _backtestSummary = [];
  bool _isLoading = false;
  String? _error;
  DateTime? _lastRefresh;

  // Getters
  DailyNewspaper? get newspaper => _newspaper;
  List<BacktestSummary> get backtestSummary => _backtestSummary;
  bool get isLoading => _isLoading;
  String? get error => _error;
  DateTime? get lastRefresh => _lastRefresh;
  
  bool get hasData => _newspaper != null;
  bool get hasError => _error != null;

  // Load daily newspaper data
  Future<void> loadDailyNewspaper() async {
    _setLoading(true);
    _clearError();

    try {
      _newspaper = await ApiService.getDailyNewspaper();
      _lastRefresh = DateTime.now();
      notifyListeners();
    } catch (e) {
      _setError('Failed to load daily newspaper: ${e.toString()}');
    } finally {
      _setLoading(false);
    }
  }

  // Load backtest summary data
  Future<void> loadBacktestSummary() async {
    try {
      _backtestSummary = await ApiService.getBacktestSummary();
      notifyListeners();
    } catch (e) {
      // Don't set main error state for backtest failure
      if (kDebugMode) {
        print('Failed to load backtest summary: $e');
      }
    }
  }

  // Refresh all data
  Future<void> refresh() async {
    await Future.wait([
      loadDailyNewspaper(),
      loadBacktestSummary(),
    ]);
  }

  // Check if data needs refresh (older than 5 minutes)
  bool get needsRefresh {
    if (_lastRefresh == null) return true;
    return DateTime.now().difference(_lastRefresh!).inMinutes > 5;
  }

  // Helper methods
  void _setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }

  void _setError(String error) {
    _error = error;
    notifyListeners();
  }

  void _clearError() {
    _error = null;
    notifyListeners();
  }

  // Clear all data
  void clear() {
    _newspaper = null;
    _backtestSummary = [];
    _error = null;
    _lastRefresh = null;
    _isLoading = false;
    notifyListeners();
  }
}