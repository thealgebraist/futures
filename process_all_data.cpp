#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <map>
#include <set>
#include <sstream> // Added for std::stringstream

namespace fs = std::filesystem;

// Structure to hold data for a single asset
struct AssetData {
    std::string ticker;
    std::vector<double> close_prices;
};

// Function to load close prices from a CSV, handling different formats
std::vector<double> load_close_prices_from_csv(const std::string& path) {
    std::vector<double> close_prices;
    std::ifstream file(path);
    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.empty()) continue;

        try {
            double price;
            // Determine which column is the close price
            if (path.find("returns_2y") != std::string::npos || path.find("ecoins_2y") != std::string::npos) {
                // Files from download_2y_10m_64_assets.py and ecoins_2y are: timestamp,close
                if (tokens.size() >= 2) {
                    price = std::stod(tokens[1]);
                } else {
                    continue; // Malformed line
                }
            } else if (path.find("audit") != std::string::npos) {
                // Files from data/audit/ are: timestamp,open,high,low,close,volume
                if (tokens.size() >= 5) {
                    price = std::stod(tokens[4]); // 5th column (index 4) is close
                } else {
                    continue; // Malformed line
                }
            } else {
                // Default to assuming it's the second column if not recognized
                if (tokens.size() >= 2) {
                    price = std::stod(tokens[1]);
                } else {
                    continue; // Malformed line
                }
            }
            if (price > 0) { // Only add valid positive prices
                close_prices.push_back(price);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line in " << path << ": " << line << " - " << e.what() << std::endl;
        }
    }
    return close_prices;
}


int main() {
    std::vector<std::string> assets = {
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
        "TRXUSDT", "DOTUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT", "UNIUSDT", "BCHUSDT",
        "NEARUSDT", "FETUSDT", "APTUSDT", "ICPUSDT", "STXUSDT", "OPUSDT", "FILUSDT",
        "XLMUSDT", "AAVEUSDT", "GRTUSDT", "HBARUSDT", "KASUSDT", "SUIUSDT", "ARBUSDT",
        "RENDERUSDT", "PEPEUSDT", "WLDUSDT", "IMXUSDT", "INJUSDT", "TIAUSDT", "LDOUSDT",
        "MKRUSDT", "KNCUSDT", "CRVUSDT", "MANAUSDT", "EGLDUSDT", "ENJUSDT", "CHZUSDT",
        "ZILUSDT", "SNXUSDT", "HOTUSDT", "DYDXUSDT", "FLOWUSDT", "IOSTUSDT", "IOTAUSDT",
        "QTUMUSDT", "RAYUSDT", "SXPUSDT", "THETAUSDT", "VETUSDT", "SCUSDT", "ONDOUSDT",
        "ONEUSDT", "ONTUSDT", "SUSHIUSDT", "ALGOUSDT", "DGBUSDT", "ALPHAUSDT", "ANKRUSDT",
        "GLMUSDT"
    };

    std::ofstream returns_out("data/all_returns_2y_10m.csv");
    returns_out << "ticker,return" << std::endl;

    std::ofstream second_diffs_out("data/all_second_diffs_2y_10m.csv");
    second_diffs_out << "ticker,second_diff" << std::endl;

    std::set<std::string> processed_assets; // To avoid processing the same asset multiple times if it exists in multiple sources

    for (const auto& ticker : assets) {
        if (processed_assets.count(ticker)) continue;

        std::string path;
        // Prioritize data/returns_2y (from new download script)
        if (fs::exists("data/returns_2y/" + ticker + "_10m.csv")) {
            path = "data/returns_2y/" + ticker + "_10m.csv";
        } 
        // Then data/ecoins_2y (from previous tasks)
        else if (fs::exists("data/ecoins_2y/" + ticker + "_10m_2y.csv")) {
            path = "data/ecoins_2y/" + ticker + "_10m_2y.csv";
        } 
        // Then data/audit (for older, shorter data if 2y data is not available)
        else if (fs::exists("data/audit/" + ticker + "/10m.csv")) {
            path = "data/audit/" + ticker + "/10m.csv";
        } else {
            std::cerr << "Warning: Data not found for " << ticker << ". Skipping." << std::endl;
            continue;
        }

        std::vector<double> close_prices = load_close_prices_from_csv(path);
        if (close_prices.empty()) {
            std::cerr << "Warning: No valid close prices loaded for " << ticker << " from " << path << ". Skipping." << std::endl;
            continue;
        }

        std::cout << "Processing " << ticker << " from " << path << ". Prices: " << close_prices.size() << std::endl;

        // Calculate log-returns
        std::vector<double> returns;
        for (size_t i = 1; i < close_prices.size(); ++i) {
            double r = std::log(close_prices[i] / close_prices[i-1]);
            if (std::isfinite(r)) {
                returns.push_back(r);
                returns_out << ticker << "," << std::fixed << std::setprecision(12) << r << "\n";
            }
        }

        // Calculate second-order differences (derivative of returns)
        std::vector<double> second_diffs;
        if (returns.size() > 1) {
            for (size_t i = 1; i < returns.size(); ++i) {
                double sd = returns[i] - returns[i-1];
                if (std::isfinite(sd)) {
                    second_diffs.push_back(sd);
                    second_diffs_out << ticker << "," << std::fixed << std::setprecision(12) << sd << "\n";
                }
            }
        }
        processed_assets.insert(ticker); // Mark as processed
    }

    returns_out.close();
    second_diffs_out.close();

    return 0;
}
