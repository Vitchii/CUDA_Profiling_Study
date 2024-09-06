#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>

class PrimeCalculator {
private:
    unsigned int limit;
    std::vector<unsigned int> primes;
    std::atomic<unsigned int> primeCount;
    double elapsedTime;
    unsigned int numThreads;
    std::mutex primesMutex;

    // Classic Primality Test
    void classicPrimalityTest() {
        auto start = std::chrono::high_resolution_clock::now();

        primes.clear();
        primeCount = 0;

        // Hardcoded primes for small numbers
        if (limit >= 2) {
            primes.push_back(2);
            ++primeCount;
        }
        if (limit >= 3) {
            primes.push_back(3);
            ++primeCount;
        }

        for (unsigned int n = 5; n <= limit; n += 6) {
            if (isPrime(n)) {
                primes.push_back(n);
                ++primeCount;
            }
            if (n + 2 <= limit && isPrime(n + 2)) {
                primes.push_back(n + 2);
                ++primeCount;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration<double>(end - start).count();
    }

    // Sieve of Eratosthenes
    void sieveOfEratosthenes() {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<bool> sieve(limit + 1, true);
        sieve[0] = sieve[1] = false;

        for (unsigned int p = 2; p * p <= limit; ++p) {
            if (sieve[p]) {
                for (unsigned int i = p * p; i <= limit; i += p) {
                    sieve[i] = false;
                }
            }
        }

        primes.clear();
        primeCount = 0;
        for (unsigned int p = 2; p <= limit; ++p) {
            if (sieve[p]) {
                primes.push_back(p);
                ++primeCount;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration<double>(end - start).count();
    }

    // Multithreaded Sieve of Eratosthenes
    void multithreadedSieveOfEratosthenes() {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<bool> sieve(limit + 1, true);
        sieve[0] = sieve[1] = false;

        unsigned int chunkSize = (limit + numThreads - 1) / numThreads; // Calculate chunk size

        std::vector<std::thread> threads;

        for (unsigned int t = 0; t < numThreads; ++t) {
            unsigned int startRange = t * chunkSize;
            unsigned int endRange = std::min(startRange + chunkSize, limit + 1);

            threads.emplace_back([this, &sieve, startRange, endRange]() {
                for (unsigned int p = 2; p * p <= limit; ++p) {
                    if (sieve[p]) {
                        for (unsigned int i = std::max(p * p, (startRange + p - 1) / p * p); i < endRange; i += p) {
                            sieve[i] = false;
                        }
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        primes.clear();
        primeCount = 0;
        for (unsigned int p = 2; p <= limit; ++p) {
            if (sieve[p]) {
                primes.push_back(p);
                ++primeCount;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration<double>(end - start).count();
    }

    static bool isPrime(unsigned int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        for (unsigned int i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) return false;
        }
        return true;
    }

    void printPrimes() const {
        std::cout << "Primes found:\n";
        for (const unsigned int prime : primes) {
            std::cout << prime << ' ';
        }
        std::cout << std::endl;
    }

public:
    explicit PrimeCalculator(unsigned int lim) : limit(lim), primeCount(0), elapsedTime(0) {
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 2; // Fallback if hardware_concurrency() returns 0
    }

    void run(int methodChoice) {
        switch (methodChoice) {
            case 1:
                std::cout << "Starting Classic Primality Test with upper limit " << limit << std::endl;
                classicPrimalityTest();
                break;
            case 2:
                std::cout << "Starting Sieve of Eratosthenes with upper limit " << limit << std::endl;
                sieveOfEratosthenes();
                break;
            case 3:
                std::cout << "Starting Multithreaded Sieve of Eratosthenes with upper limit " << limit << std::endl;
                multithreadedSieveOfEratosthenes();
                break;
            case 4:
                std::cout << "Starting Classic Primality Test with upper limit " << limit << std::endl;
                classicPrimalityTest();
                std::cout << "Number of primes found: " << primeCount << std::endl;
                std::cout << "Execution time: " << std::fixed << std::setprecision(6) << elapsedTime << " seconds"
                    << std::endl;
                std::cout << std::endl;

                std::cout << "Starting Sieve of Eratosthenes with upper limit " << limit << std::endl;
                sieveOfEratosthenes();
                std::cout << "Number of primes found: " << primeCount << std::endl;
                std::cout << "Execution time: " << std::fixed << std::setprecision(6) << elapsedTime << " seconds"
                    << std::endl;
                std::cout << std::endl;

                std::cout << "Starting Multithreaded Sieve of Eratosthenes with upper limit " << limit << std::endl;
                multithreadedSieveOfEratosthenes();
                std::cout << "Number of primes found: " << primeCount << std::endl;
                std::cout << "Execution time: " << std::fixed << std::setprecision(6) << elapsedTime << " seconds"
                    << std::endl;
                std::cout << std::endl;
                break;
            default:
                std::cerr << "Invalid choice" << std::endl;
                return;
        }

        if (methodChoice != 4) {
            std::cout << "Number of primes found: " << primeCount << std::endl;
            std::cout << "Execution time: " << std::fixed << std::setprecision(6) << elapsedTime << " seconds"
                << std::endl;

            if (primeCount < 32 || (primeCount >= 32 && promptForPrimes())) {
                printPrimes();
            }
        }
    }

    static bool promptForPrimes() {
        char choice;
        std::cout << "Do you want to see the list of primes? (y/n): ";
        std::cin >> choice;
        return choice == 'y' || choice == 'Y';
    }
};

int main() {
    int methodChoice;
    std::cout << "Select method: 1 = Prime Test; 2 = Sieve of Eratosthenes; "
                 "3 = Multithreaded Sieve of Eratosthenes; 4 = All\n";
    std::cin >> methodChoice;
    std::cout << std::endl;

    int limitChoice;
    unsigned int limit = 0;
    std::cout << "Select upper bound: 1 = 1,000; 2 = 100,000,000; "
                 "3 = 1,000,000,000; 4 = 4,000,000,000; 5 = Custom upper bound\n";
    std::cin >> limitChoice;

    switch (limitChoice) {
        case 1: limit = 1000; break;
        case 2: limit = 100000000; break;
        case 3: limit = 1000000000; break;
        case 4: limit = 4000000000; break;
        case 5:
            std::cout << "Enter the custom upper bound: ";
            std::cin >> limit;
            break;
        default:
            std::cerr << "Invalid choice, exiting." << std::endl;
            return 1;
    }
    std::cout << std::endl;

    PrimeCalculator calculator(limit);
    calculator.run(methodChoice);

    return 0;
}
