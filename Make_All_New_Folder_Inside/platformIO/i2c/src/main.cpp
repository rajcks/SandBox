#include <Arduino.h>
#include "Adafruit_TCS34725.h"
 
class ArdudinoFunctions {
  private:
 
    Adafruit_TCS34725 tcs;
 
    // Helper function: read 10 ADC values and return the average
    int readADCAverage(uint8_t pin) {
      long sum = 0;
      for (int i = 0; i < 10; i++) {
        sum += analogRead(pin);
      }
      return sum / 10;
    }
 
    // Helper function: trigger digital output ON, return true if successful
    bool setDigitalOn(uint8_t pin) {
      digitalWrite(pin, HIGH);
      return digitalRead(pin) == HIGH;
    }
 
    // Helper function: trigger digital output OFF, return true if successful
    bool setDigitalOff(uint8_t pin) {
      digitalWrite(pin, LOW);
      return digitalRead(pin) == LOW;
    }
 
    // Helper function: get the RGB value from the color sensor
    void readColor(float &r, float &g, float &b) {
      uint16_t clear, red, green, blue;
      tcs.getRawData(&red, &green, &blue, &clear);
 
      uint32_t sum = clear;
      if (sum == 0) sum = 1;  // prevent division by zero
 
      r = (float)red / sum * 256;
      g = (float)green / sum * 256;
      b = (float)blue / sum * 256;
    }
 
  public:
    // Constructor: configure ADC & digital pins
    ArdudinoFunctions():
      tcs(TCS34725_INTEGRATIONTIME_154MS, TCS34725_GAIN_4X)
    {
      pinMode(A1, INPUT);
      pinMode(A2, INPUT);
      pinMode(A3, INPUT);
      pinMode(A4, INPUT);
 
      pinMode(3, OUTPUT);
      pinMode(4, OUTPUT);
      pinMode(6, OUTPUT);
      pinMode(7, OUTPUT);
 
      // Initialize TCS34725
      if (!tcs.begin()) {
        Serial.println("No TCS34725 found ... check your connections.");
        while (1); // Halt
      }
    }
 
    // ----------- ADC functions -------------
    int readVminus() { return readADCAverage(A3); }
    int readVplus()  { return readADCAverage(A4); }
    int readADC_PEGEL() { return readADCAverage(A1); }
    int readADC_FFT()   { return readADCAverage(A2); }
 
    // ----------- Digital control ON -------------
    bool VminusOn() { return setDigitalOn(3); }
    bool VplusOn()  { return setDigitalOn(4); }
    bool TxOn()     { return setDigitalOn(6); }
    bool RxOn()     { return setDigitalOn(7); }
 
    // ----------- Digital control OFF -------------
    bool VminusOff() { return setDigitalOff(3); }
    bool VplusOff()  { return setDigitalOff(4); }
    bool TxOff()     { return setDigitalOff(6); }
    bool RxOff()     { return setDigitalOff(7); }
 
    // ----------- Color detection -------------
    bool isRed() {
      float r, g, b;
      readColor(r, g, b);
      return (r > g && r > b);  // dominant red
    }
 
    bool isGreen() {
      float r, g, b;
      readColor(r, g, b);
      return (g > r && g > b);  // dominant green
    }
 
    bool isBlue() {
      float r, g, b;
      readColor(r, g, b);
      return (b > r && b > g);  // dominant blue
    }
};

ArdudinoFunctions controller;

void setup() {
  Serial.begin(9600);
  
  // Test: Turn V- ON
  controller.VminusOn();
}
 
void loop() {
  if (controller.isRed()) {
    Serial.println("Detected: RED");
  } else if (controller.isGreen()) {
    Serial.println("Detected: GREEN");
  } else if (controller.isBlue()) {
    Serial.println("Detected: BLUE");
  }
 
  int val = controller.readVplus();
  Serial.println(val);
 
  delay(1000);
}