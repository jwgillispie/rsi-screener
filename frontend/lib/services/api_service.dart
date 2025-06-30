import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../models/daily_newspaper.dart';
import '../models/iron_condor_signal.dart';
import '../models/backtest_summary.dart';

class ApiService {
  // Change this to your computer's IP address when testing on mobile
  // For iOS simulator/Android emulator, localhost should work
  static const String baseUrl = 'http://localhost:8000';
  
  static const Duration timeoutDuration = Duration(seconds: 30);

  static Future<DailyNewspaper> getDailyNewspaper() async {
    try {
      final response = await http
          .get(
            Uri.parse('$baseUrl/daily-newspaper'),
            headers: {'Content-Type': 'application/json'},
          )
          .timeout(timeoutDuration);

      if (response.statusCode == 200) {
        final Map<String, dynamic> json = jsonDecode(response.body);
        return DailyNewspaper.fromJson(json);
      } else {
        throw ApiException(
          'Failed to load daily newspaper: ${response.statusCode}',
          response.statusCode,
        );
      }
    } on SocketException {
      throw ApiException(
        'No internet connection. Make sure the API server is running on $baseUrl',
        0,
      );
    } on http.ClientException {
      throw ApiException(
        'Connection failed. Check if the API server is running.',
        0,
      );
    } catch (e) {
      throw ApiException('Unexpected error: $e', 0);
    }
  }

  static Future<List<IronCondorSignal>> getSignals() async {
    try {
      final response = await http
          .get(
            Uri.parse('$baseUrl/signals'),
            headers: {'Content-Type': 'application/json'},
          )
          .timeout(timeoutDuration);

      if (response.statusCode == 200) {
        final Map<String, dynamic> json = jsonDecode(response.body);
        final List<dynamic> signalsJson = json['signals'];
        return signalsJson
            .map((signalJson) => IronCondorSignal.fromJson(signalJson))
            .toList();
      } else {
        throw ApiException(
          'Failed to load signals: ${response.statusCode}',
          response.statusCode,
        );
      }
    } on SocketException {
      throw ApiException(
        'No internet connection. Make sure the API server is running.',
        0,
      );
    } catch (e) {
      throw ApiException('Unexpected error: $e', 0);
    }
  }

  static Future<List<BacktestSummary>> getBacktestSummary() async {
    try {
      final response = await http
          .get(
            Uri.parse('$baseUrl/backtest-summary'),
            headers: {'Content-Type': 'application/json'},
          )
          .timeout(timeoutDuration);

      if (response.statusCode == 200) {
        final List<dynamic> json = jsonDecode(response.body);
        return json
            .map((summaryJson) => BacktestSummary.fromJson(summaryJson))
            .toList();
      } else {
        throw ApiException(
          'Failed to load backtest summary: ${response.statusCode}',
          response.statusCode,
        );
      }
    } on SocketException {
      throw ApiException(
        'No internet connection. Make sure the API server is running.',
        0,
      );
    } catch (e) {
      throw ApiException('Unexpected error: $e', 0);
    }
  }

  static Future<Map<String, dynamic>> getConfig() async {
    try {
      final response = await http
          .get(
            Uri.parse('$baseUrl/config'),
            headers: {'Content-Type': 'application/json'},
          )
          .timeout(timeoutDuration);

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw ApiException(
          'Failed to load config: ${response.statusCode}',
          response.statusCode,
        );
      }
    } on SocketException {
      throw ApiException(
        'No internet connection. Make sure the API server is running.',
        0,
      );
    } catch (e) {
      throw ApiException('Unexpected error: $e', 0);
    }
  }

  static Future<Map<String, dynamic>> triggerBacktest({
    int nStocks = 10,
    int startYear = 2024,
  }) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl/run-backtest?n_stocks=$nStocks&start_year=$startYear'),
            headers: {'Content-Type': 'application/json'},
          )
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw ApiException(
          'Failed to trigger backtest: ${response.statusCode}',
          response.statusCode,
        );
      }
    } on SocketException {
      throw ApiException(
        'No internet connection. Make sure the API server is running.',
        0,
      );
    } catch (e) {
      throw ApiException('Unexpected error: $e', 0);
    }
  }

  static Future<bool> checkHealth() async {
    try {
      final response = await http
          .get(
            Uri.parse('$baseUrl/health'),
            headers: {'Content-Type': 'application/json'},
          )
          .timeout(const Duration(seconds: 5));

      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}

class ApiException implements Exception {
  final String message;
  final int statusCode;

  ApiException(this.message, this.statusCode);

  @override
  String toString() => 'ApiException: $message';
}