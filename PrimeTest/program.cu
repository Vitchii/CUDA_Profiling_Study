/* 
CUDA-Programm zur Veranschaulichung der Herauforderungen des Profilings im Kontext der GPGPU-Programmierung
Bachelorarbeit Informatik, Universität Trier, 2024
Fabian Vecellio del Monego, 2024
*/

// ###################################################################################################################

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>
#include <string>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>

using int64 = long long int;  // weil "long" unter Windows nur 32 Bit hat und "long long int" unübersichtlich ist
using int32u = unsigned int;

// CUDA-Kernel #######################################################################################################

// Trivialer Primzahl-Test
__global__ void primeTest1(bool* array, int64 l, int64 u) {
    int64 index = threadIdx.x + blockIdx.x * blockDim.x; // Index der Zahl finden, die der Thread bearbeitet

    if (index >= l && index <= u) { // Range "durchlaufen"
        if (index > 1) {
            for (long i = 2; i <= sqrtf(index); i++) { // Suche nach Teiler bis zur Quadratwurzel der Zahl
                if (index % i == 0) {  // wenn die Zahl durch eine andere Zahl teilbar ist, ist sie keine Primzahl
                    array[index - l] = false;
                }
            }
        }
        else array[index - l] = false; // 0 und 1 sind keine Primzahlen
    }
    return;
}

// Performanterer Test
__global__ void primeTest2(bool* array, int64 l, int64 u) { 
    int64 index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= l && index <= u) {
        if (index > 2 && index % 2 != 0) { // Nur ungerade Zahlen prüfen
            for (long i = 3; i <= sqrtf(index); i += 2) { // Nur ungerade Teiler prüfen
                if (index % i == 0) {
                    array[index - l] = false;
                }
            }
        }
        else if (index == 2) {
            array[index - l] = true; // 2 ist eine Primzahl
        }
        else array[index - l] = false; // 0, 1 und gerade Zahlen sind keine Primzahlen
    }
    return;
}

// Noch performanterer Test
__global__ void primeTest3(bool* array, int64 l, int64 u) {
    int64 index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index == 2 || index == 3 || index > u) { // 2 und 3 sind Primzahlen
        return;
    }
    if (index <= 1 || index % 2 == 0 || index % 3 == 0) { // 0, 1 und Vielfache von 2 und 3 sind keine Primzahlen
        array[index - l] = false;
        return;
    }
    for (long i = 5; i <= sqrtf(index); i += 6) { // Primzahlen größer 3 folgen dem Muster 6k ± 1
        if (index % i == 0 || index % (i + 2) == 0)
            array[index - l] = false;
    }
    return;
}

// Noch performanterer Test, unsigned int, bis 4 Mrd.
__global__ void primeTest3Unsigned(bool* array, int32u u) { 
    int32u index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index == 2 || index == 3 || index > u) { // 2 und 3 sind Primzahlen
        return;
    }
    if (index <= 1 || index % 2 == 0 || index % 3 == 0) { // 0, 1 und Vielfache von 2 und 3 sind keine Primzahlen
        array[index] = false;
        return;
    }
    for (int32u i = 5; i <= sqrtf(index); i += 6) { // Primzahlen größer 3 folgen dem Muster 6k ± 1
        if (index % i == 0 || index % (i + 2) == 0)
            array[index] = false;
    }
    return;
}

// Noch performanterer Test, unsigned int, nur ungerade Zahlen
__global__ void primeTest3UnsignedOdd(bool* array, int32u u) {
    int32u index = threadIdx.x + blockIdx.x * blockDim.x;
    int32u i = index * 2 - 1; // Index anpassen gemäß 2i - 1, sodass i immer ungerade ist

    if (i == 1 || (i != 3 && i % 3 == 0)) { // 1 und Vielfache von 3 sind keine Primzahlen
        array[index] = false;
        return;
    }
    for (int32u j = 5; j <= sqrtf(i); j += 6) { // Primzahlen größer 3 folgen dem Muster 6k ± 1
        if (i % j == 0 || i % (j + 2) == 0) {
            array[index] = false;
            return;
        }
    }
    return;
}

// Noch performanterer Test, unsigned int, nur ungerade Zahlen
__global__ void primeTest3Inverted(bool* array, int32u u) {
    int32u index = threadIdx.x + blockIdx.x * blockDim.x;
    int32u i = index * 2 - 1; // Index anpassen gemäß 2i - 1, sodass i immer ungerade ist

    if (i == 1 || (i != 3 && i % 3 == 0)) { // 1 und Vielfache von 3 sind keine Primzahlen
        return;
    }
    for (int32u j = 5; j * j <= i; j += 6) { // Primzahlen größer 3 folgen dem Muster 6k ± 1
        if (i % j == 0 || i % (j + 2) == 0) {
            return;
        }
    }
    array[index] = false;
    return;
}

// Sieb des Eratosthenes
__global__ void sieveEratosthenes(bool* array, int64 u) { 
    int64 index = threadIdx.x + blockIdx.x * blockDim.x;

    if (array[index]) { // nur berechnen, wenn die Zahl noch nicht ausgeschlossen wurde
        if (index >= 2) {
            int64 i = index * index; // Noch nicht gefundene Primzahlen können nur größer als index^2 sein

            for (i; i <= u; i += index) { // Alle Vielfache als nicht prim markieren
                array[i] = false;
            }
        }
        else if (index == 0 || index == 1) { // 0 und 1 sind keine Primzahlen
            array[index] = false;
        }
    }
    return;
}

// Sieb des Eratosthenes, unsigned int, bis 4 Mrd.
__global__ void sieveEratosthenesUnsigned(bool* array, int32u u) {
    int32u index = threadIdx.x + blockIdx.x * blockDim.x;

    if (array[index]) { // nur berechnen, wenn die Zahl noch nicht ausgeschlossen wurde
        if (index >= 2 && index <= sqrtf(u)) { // Test, um nicht über 32 Bit hinaus zu rechnen
            int32u i = index * index; // Noch nicht gefundene Primzahlen können nur größer als index^2 sein

            for (i; i <= u; i += index) { // Alle Vielfache als nicht prim markieren
                array[i] = false;
            }
        }
        else if (index == 0 || index == 1) { // 0 und 1 sind keine Primzahlen
            array[index] = false;
        }
    }
    return;
}

__global__ void sieveEratosthenesUnsignedOdd(bool* array, int32u u) {
    int32u index = threadIdx.x + blockIdx.x * blockDim.x;

    // Nur ungerade Zahlen werden berücksichtigt
    int32u i = 2 * index + 1;

    if (index > 0 && i * i <= u * 2) {
        if (array[index]) {
            for (int32u j = i * i; j <= u * 2; j += i * 2) {
                array[(j / 2) + 1] = false;
            }
        }
    }
    return;
    /* Leider funktioniert dieser Kernel nicht korrekt.Im Bereich bis bspw. 1000 haut es noch hin, aber bei größeren 
       Intervallen werden nicht mehr alle Primzahlen gefunden. Desto größer das Intervall, desto mehr Primzahlen 
       bleiben unerkannt. Ich habe einige Zeit in das Debugging dieses Kernels gesteck, mich aufgrund der eigentlichen 
       Zielsetzugn dieser Arbeit und der ohnehin performanteren Methode primeTest3UnsignedOdd dagegen entschieden, 
       weiter Zeit in diese eigentlich sehr interessante Variation des Siebes des Eratosthenes zu stecken. Ich habe 
       den Kernel aber dennoch im Code belassen - vielleicht finde ich ja später einmal eine Lösung. */
}

// Sieb des Sundaram
__global__ void sieveSundaram(bool* array, int64 u) {
    int64 index = threadIdx.x + blockIdx.x * blockDim.x + 1; // Sieb funktioniert nicht mit 0, daher + 1

    if (index < u) { // nur berechnen, wenn die Zahl in der Range liegt
        int64 n = (u - 2) / 2; // obere Grenze anpassen

        for (int64 i = index; (index + i + 2 * index * i) <= n; i++) { // Iterieren bis (Obergrenze - 2) / 2
            array[index + i + 2 * index * i] = false; // Markieren der Zahlen i, sodass 2 * i + 1 != prim
        }
    }
    return;
}

// Test-Kernel
__global__ void primeTestDebug(bool* array, int32u n) {
    int64 i = threadIdx.x + blockIdx.x * blockDim.x + 1;

    for (int64 j = i; (i + j + 2 * i * j) <= n; j++) {
        array[i + j + 2 * i * j] = false;
    }
    return;
}

// Hilfsfunktionen ###################################################################################################

void clearInputBuffer() { // Hilfsfunktion, um den Eingabepuffer zu leeren
    int c;
    while ((c = getchar()) != '\n' && c != EOF) {}
    return;
}

void askMethod(int& method, std::string& kernelString) { // Funktion, um die Methode zu wählen
    int option;
    printf("Please choose a method. (1, 2, 3, 4, 5)\n");
    printf("1 = Trivial Test; 2 = Enhanced Test; 3 = Further enhanced Test; "
        "4 = Sieve of Eratosthenes; 5 = Sieve of Sundaram\n");
    printf("Enter method option: ");
    clearInputBuffer;
    scanf_s("%d", &option);

    switch (option) {
        case 0:     { method = option; kernelString = "DEBUG KERNEL";                 break; }
        case 1:     { method = option; kernelString = "primeTest1";                   break; }
        case 2:     { method = option; kernelString = "primeTest2";                   break; }
        case 3:     { method = option; kernelString = "primeTest3";                   break; }
        case 4:     { method = option; kernelString = "sieveEratosthenes";            break; }
        case 5:     { method = option; kernelString = "sieveSundaram";                break; }
        case 33:    { method = option; kernelString = "primeTest3Unsigned";           break; }
        case 44:    { method = option; kernelString = "sieveEratosthenesUnsigned";    break; }
        case 333:   { method = option; kernelString = "primeTest3UnsignedOdd";        break; }
        case 444:   { method = option; kernelString = "sieveEratosthenesUnsignedOdd"; break; }
        case 3333:  { method = option; kernelString = "primeTest3Inverted";           break; }
        default: {
            printf("Invalid option. Will commence with the default method: Trivial Test.\n");
            method = 1; // Standardfall ist der triviale Test
            kernelString = "primeTest1";
            break;
        }
    }

    printf("\n");
    return;
}

void askRange(int64& l, int64& u, int method) { // Funktion, um das zu prüfende Intervall zu wählen
    int option;
    printf("Please choose an upper bound. (1, 2, 3, 4, 5)\n");
    if (method <= 3) { // nur diese Methode erlauben l > 0
        printf("1 = 1,000; 2 = 100,000,000; 3 = 1,000,000,000; 4 = 4,000,000,000; 5 = custom range\n");
    }
    else {
        printf("1 = 1,000; 2 = 100,000,000; 3 = 1,000,000,000; 4 = 4,000,000,000; 5 = custom upper bound\n");
    }
    printf("Enter range option: ");
    clearInputBuffer;
    scanf_s("%d", &option);

    l = 0; // Standardwert für untere Grenze ist 0
    switch (option) {
        // Test-Option
        case 0: {
            u = 10;
            break;
        }
        // Reguläre Optionen
        case 1: {
            u = 1000;
            break;
        }
        case 2: {
            u = 100000000; // 100 Mio.
            break;
        }
        case 3: {
            u = 1000000000; // 1 Mrd.
            break;
        }
        case 4: {
            u = 4000000000; // 4 Mrd.
            break;
        }
        case 5: {
            printf("\nPlease define the range of natural numbers you want to check for primes.\n");
            if (method <= 3 || method == 33) { // nur diese Methode erlauben l > 0
                printf("Enter lower bound: ");
                clearInputBuffer;
                scanf_s("%lld", &l);
            }
            printf("Enter upper bound: ");
            clearInputBuffer;
            scanf_s("%lld", &u);
            if (l > u) { // wenn die obere Grenze  kleiner als die untere ist
                printf("Invalid range. Will commence with default bound: 50000000.\n");
                l = 0, u = 1000000;
            }
            if (l < 0) { // wenn die untere Grenze negativ ist
                l = 0;
                if (u < l) u = l; // wenn die obere Grenze jetzt kleiner als die untere ist
                printf("Adjusting range to natural numbers, since prime numbers are positive by definition.\n"
                    "New Range: %lld to %lld.\n", l, u);
            }
            if ((method == 33 || method == 333 || method == 3333 || 
                method == 44 || method == 444) && u >= 4294311961) {
                u = 4294311960; // maximaler Wert, bei dem die Schleife in primeTest3333 terminieren kann
                printf("This method can't handle numbers that big.\n"
                    "New Range: %lld to %lld.\n", l, u);
		    }
            break;
        }
        // Test-Optionen
        case 6: {
            u = 10000000; // 10 Mio.
            break;
        }
        case 7: {
            u = 2000000000; // 2 Mrd.
            break;
        }
        case 8: {
            u = 8000000000; // 8 Mrd.
            break;
        }
        case 9: {
            l = 10;
            u = 1000;
            break;
        }
        default: {
            u = 1000000; // Standardwert: 1 Million
            printf("Invalid option. Will commence with the default bound: %lld.\n", u);
        }
    }
    printf("\n");
    return;
}

void askBlockSize(int& blockSize) { // Funktion, um die Blockgröße zu wählen oder alle zu testen
    char input[10];
    printf("Please choose the amount of threads per block. (32, 64, 128, 256, 512, 1024, all)\n");
    printf("Enter block size: ");
    clearInputBuffer();
    fgets(input, sizeof(input), stdin);

    input[strcspn(input, "\n")] = 0; // Eingabe bereinigen

    if (strcmp(input, "all") == 0) {
        blockSize = 1;
        return;
    }

    bool isValidNumber = true; // Prüfen, ob die Eingabe eine Zahl ist
    for (int i = 0; input[i] != '\0'; i++) {
        if (!isdigit(input[i])) {
            isValidNumber = false;
            break;
        }
    }

    if (isValidNumber) {
        int option = atoi(input);
        if (option % 32 == 0 && option <= 1024) {
            blockSize = option;
            return;
        }

        if (option >= 1 && option <= 6) {
            switch (option) {
                case 1: { blockSize = 32;   return; }
                case 2: { blockSize = 64;   return; }
                case 3: { blockSize = 128;  return; }
                case 4: { blockSize = 256;  return; }
                case 5: { blockSize = 512;  return; }
                case 6: { blockSize = 1024; return; }
            }
            return;
        }
    }

    printf("Invalid option. Will commence with the default block size of 256.\n");
    blockSize = 256; // Standardfall
    return;
}

void output(bool* array, int64 l, int64 u, int64 rangeSize, long primeCount) { // Ausgabe-Funktion
    if (primeCount > 0) {
        std::string input = "y"; // Ausgabe der gefunden Primzahlen ist standardmäßig aktiviert

        printf("\n");
        if (primeCount > 25) { // keine automatische Ausgabe, wenn mehr als x Primzahlen gefunden wurden
            printf("Dou you want a list? (y/n)\n");
            printf("Enter y or n: ");
            std::getline(std::cin, input);
        }

        if (input == "y" || input == "Y") {
            // printf("\nPrimes in range %d to %d:\n", l, u);
            int lineLength = 0;
            std::ostringstream oss;
            std::string primeString;
            printf("\n");
            for (int64 i = 0; i < rangeSize; i++) { // Ausgabe der Primzahlen
                if (array[i]) { // wenn die Zahl eine Primzahl ist
                    oss.str("");
                    oss << i;
                    primeString = oss.str();
                    if (lineLength + primeString.length() + 1 > 120) { // Prüfen, 
                        // ob das Hinzufügen der nächsten Zahl die maximale Zeilenlänge überschreitet
                        std::cout << std::endl;  // Zeilenumbruch
                        lineLength = 0;   // Zähler zurücksetzen
                    }
                    std::cout << primeString << " ";
                    lineLength += primeString.length() + 1; // Aktualisieren der aktuellen Zeilenlänge
                }
            }
            printf("\n");
        }
        else {
            printf("\nOkay, bye!\n");
        }
    }
    return;
}

// Hauptfunktion #####################################################################################################

int main() {
    // Methode wählen
    int method;
    std::string kernelString;
    askMethod(method, kernelString);
    bool needsOddTransformation = false; // ob nur ungerade Zahlen geprüft werden
    bool needsSundaramTransformation = false; // ob das Array transformiert werden muss
    bool needsInversion = false; // ob das Array invertieert werden muss

    // Welche Zahlen sollen geprüft werden?
    int64 l, u; // untere und obere Grenze
    askRange(l, u, method);
    int64 rangeSize = u + 1 - l;
    int64 arraySize = rangeSize;
    int32u n = (u - 2) / 2; // für Sundaram-Test-Kernel

    if (method >= 100) { // für Methoden, die nur ungerade Zahlen prüfen
        arraySize = rangeSize / 2; // Array-Größe anpassen
        needsOddTransformation = true;
    }

    // Parameter für den Kernel setzen
    int blockSize; // Anzahl der Threads pro Block
    askBlockSize(blockSize);

    bool testAll = false;
    double times[6]; // Zeiten für die verschiedenen Blockgrößen
    if (blockSize == 1) { // alle Blockgrößen testen
        testAll = true;
        blockSize = 32;
	}

    // ==============================================================================================================

    printf("\nInitializing arrays ... ");

    // Array auf dem Host anlegen und initialisieren
    bool* array;
    array = (bool*)malloc(arraySize * sizeof(bool)); // Array auf dem Host anlegen
    for (int64 i = 0; i < arraySize; i++) { // Array mit true initialisieren
        array[i] = true;
    }

    for (int run = 0; run < 6; run++) { // Schleife, wenn alle Blockgrößen getestet werden

        // Array auf dem Device anlegen und kopieren
        bool* deviceArray;
        cudaError_t error; // Fehler-Variable
        error = cudaMalloc((void**)&deviceArray, arraySize * sizeof(bool));
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error in cudaMalloc: %s\n", cudaGetErrorString(error));
        }
        error = cudaMemcpy(deviceArray, array, arraySize * sizeof(bool), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error in cudaMemcpy (HostToDevice): %s\n", cudaGetErrorString(error));
        }

        // Kernel-Konfiguration berechnen
        int blocks = (arraySize + blockSize - 1) / blockSize; // Anzahl der Blöcke berechnen,
        // sodass immer genug Threads für alle Zahlen vorhanden sind (in 256er Schritten)

        printf("\nLaunching CUDA kernel %s with %d threads per block ... ", kernelString.c_str(), blockSize);
        if (!testAll) printf("\nSearching for primes in range %lld to %lld ... ", l, u);
        cudaProfilerStart;

        // Zeitmessung starten
        auto start = std::chrono::high_resolution_clock::now();

        // Kernel starten
        switch (method) {
            case 0: { // Test-Kernel 
                primeTestDebug <<< blocks, blockSize >>> (deviceArray, n);
                needsSundaramTransformation = true;
                break;
            }
            case 1: { // Trivialer Test
                primeTest1 <<< blocks, blockSize >>> (deviceArray, l, u);
                break;
            }
            case 2: { // Verbesserter Test
                primeTest2 <<< blocks, blockSize >>> (deviceArray, l, u);
                break;
            }
            case 3: { // Weiter verbesserter Test
                primeTest3 <<< blocks, blockSize >>> (deviceArray, l, u);
                break;
            }
            case 4: { // Sieb des Eratosthenes
                sieveEratosthenes <<< sqrtf(blocks), blockSize >>> (deviceArray, u);
                break;
            }
            case 5: { // Sieb des Sundaram
                sieveSundaram <<< sqrtf(blocks), blockSize >>> (deviceArray, u);
                needsSundaramTransformation = true;
                break;
            }
            case 33: { // Weiter verbesserter Test mit unsigned int (32 Bit)
                int32u uTemp = u;
                primeTest3Unsigned <<< blocks, blockSize >>> (deviceArray, uTemp);
                break;
            }
            case 44: { // Sieb des Eratosthenes mit unsigned int (32 Bit)
                int32u uTemp = u;
                sieveEratosthenesUnsigned <<< sqrtf(blocks), blockSize >>> (deviceArray, uTemp);
                break;
            }
            case 333: { // Weiter verbesserter Test mit unsigned int (32 Bit), nur ungerade Zahlen
                int32u uTemp = u / 2;
                primeTest3UnsignedOdd <<< blocks, blockSize >>> (deviceArray, uTemp);
                break;
            }
            case 444: { // Weiter optimiertes Sieben mit unsigned int (32 Bit), nur ungerade Zahlen
                int32u uTemp = u / 2;
                sieveEratosthenesUnsignedOdd <<< sqrtf(blocks), blockSize >>> (deviceArray, uTemp);
                break;
            }
            case 3333: { // Noch weiter verbesserter Test
                int32u uTemp = u / 2;
                primeTest3Inverted <<< blocks, blockSize >>> (deviceArray, uTemp);
                needsInversion = true;
                break;
            }
        }

        cudaDeviceSynchronize(); // Warten, bis alle Threads fertig sind
        cudaProfilerStop;

        // Zeitmessung stoppen
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        times[run] = elapsed.count();

        if (!testAll || run == 5) { // wenn nur eine Blockgröße getestet wird oder alle getestet wurden
            // Ergebnis von Device auf Host kopieren
            error = cudaMemcpy(array, deviceArray, arraySize * sizeof(bool), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                fprintf(stderr, "CUDA error in cudaMemcpy (DeviceToHost): %s\n", cudaGetErrorString(error));
            }

            run = 6; // Schleife beenden
        }
        else {
            blockSize *= 2; // Blockgröße verdoppeln
        }

        // Speicher wieder freigeben
        error = cudaFree(deviceArray); 
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error in cudaFree: %s\n", cudaGetErrorString(error));
        }
    }

    // Ausgabe vorbereiten ==========================================================================================

    if (needsInversion) { // falls Primzahlen als false markiert wurden
        printf("\nInverting array ... ");
        bool* tempArray = (bool*)malloc(rangeSize * sizeof(bool)); // Neues Array anlegen

        for (int64 i = 0; i <= u; i++) { // Array mit false initialisieren
            tempArray[i] = false;
        }

        for (int64 i = 0; i < arraySize; i++) {
            if (!array[i]) {
				tempArray[i] = true;
			}
        }

        array = tempArray; // Pointer umleiten
    }

    if (needsOddTransformation) { // für Methoden, die nur ungerade Zahlen prüfen
        printf("\nTransforming array ... ");
        bool* tempArray = (bool*)malloc(rangeSize * sizeof(bool)); // Neues Array anlegen

        for (int64 i = 0; i <= u; i++) { // Array mit false initialisieren
            tempArray[i] = false;
        }

        for (int64 i = 2; i < arraySize; i++) { // Array transformieren gemäß 2i - 1
            tempArray[2 * i - 1] = array[i];
        }

        tempArray[2] = true; // Sonderfall 2 kann bei Methode zu ungeraden Zahlen nicht berücksichtigt werden

        array = tempArray; // Pointer umleiten
    }

    if (needsSundaramTransformation) { // für das Sundaram-Sieb
        if (!needsOddTransformation) printf("\nTransforming array ... ");
        bool* tempArray = (bool*)malloc(rangeSize * sizeof(bool)); // Neues Array anlegen

        for (int64 i = 0; i <= u; i++) { // Array mit false initialisieren
            tempArray[i] = false;
        }

        for (int64 i = 0; i <= (u - 2) / 2; i++) { // Array transformieren
            if (array[i]) {
                int64 prime = 2 * i + 1;
                if (prime == 1) { // 1 ist keine Primzahl, 2 aber schone 
                    tempArray[prime + 1] = true;
                }
                else {
                    tempArray[prime] = true;
                }
            }
        }

        std::copy(tempArray, tempArray + rangeSize, array); // Array kopieren
        free(tempArray); // Speicher freigeben
    }

    // Anzahl der Primzahlen zählen und ausgeben ====================================================================
    long primeCount = 0;
    for (int64 i = 0; i < rangeSize; i++) {
        if (array[i]) { // Anzahl der Primzahlen zählen
            primeCount++;
        }
    }

    // Ausgabe-Methode instruieren 
    printf("\n\nFound %d primes in range %lld to %lld. ", primeCount, l, u);
    if (!testAll) {
        printf("This took %lf seconds.", times[0]);
    }
    else {
        printf("\n\n%10s %10s %10s %10s %10s %10s %10s", "Block size:", "32", "64", "128", "256", "512", "1024");
        printf("\n%10s %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf \n", "Time (s):  ",
            times[0], times[1], times[2], times[3], times[4], times[5]);
    }
    output(array, l, u, rangeSize, primeCount); // Gibt Werte an Ausgabe-Funktion weiter
    
    free(array); // Speicher freigeben  

    return 0;
}

/* Ausgewählte Ergebnisse ###########################################################################################
    1.000.000.000: 
        1) 455s, 2) 234s, 3) 138s, 4) 98s, 5) 138s; 
        33) 33s, 44: 82s, 
        333) 17s, 444) 27s [falsch]
        3333) 12s, 
    4.000.000.000: 
        3) 1107s, 4) 400s,
        33) 270s, 44) 330s, 
        333) 134s 
        3333) 93s 
 */
