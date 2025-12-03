#include <SPI.h>
 
// --------------------- Pin Definitions ---------------------
#define RFM_CS 10   // Chip-select (Mega 2560 pin 53)
 
// --------------------- MikroE Constants ---------------------
#define RFMETER_FILTER_USEFULL_DATA 0x1FFF
#define RFMETER_ADC_RESOLUTION     4096.0
#define RFMETER_DEF_VREF           2.5
#define RFMETER_DEF_SLOPE          -0.025
#define RFMETER_DEF_INTERCEPT      20.0
#define RFMETER_DEF_LIMIT_HIGH     2.0
#define RFMETER_DEF_LIMIT_LOW      0.5
 
// ------------------------------------------------------------
 
// Initialize SPI & CS
void rfmeter_init() {
  pinMode(RFM_CS, OUTPUT);
  digitalWrite(RFM_CS, HIGH);
 
  SPI.begin();
  SPI.beginTransaction(SPISettings(250000, MSBFIRST, SPI_MODE0));
 
  Serial.println("-----------------------");
  Serial.println("    RF Meter Click");
  Serial.println("-----------------------");
}
 
// Read 12-bit raw ADC data
uint16_t rfmeter_read_raw() {
  digitalWrite(RFM_CS, LOW);
  delayMicroseconds(2);
 
  uint8_t high = SPI.transfer(0x00);
  uint8_t low  = SPI.transfer(0x00);
 
  digitalWrite(RFM_CS, HIGH);
  delayMicroseconds(2);
 
  uint16_t val = ((uint16_t)high << 8) | low;
  val &= RFMETER_FILTER_USEFULL_DATA;
  val >>= 1;
 
  return val;
}
 
// Convert raw data to voltage
float rfmeter_get_voltage() {
  uint16_t raw = rfmeter_read_raw();
  return (raw * RFMETER_DEF_VREF) / RFMETER_ADC_RESOLUTION;
}
 
// Convert voltage to signal strength in dBm
float rfmeter_get_signal_strength(float voltage) {
  float result;
 
  if (voltage > RFMETER_DEF_LIMIT_HIGH)
    result = (RFMETER_DEF_LIMIT_HIGH / RFMETER_DEF_SLOPE) + RFMETER_DEF_INTERCEPT;
  else if (voltage < RFMETER_DEF_LIMIT_LOW)
    result = (RFMETER_DEF_LIMIT_LOW / RFMETER_DEF_SLOPE) + RFMETER_DEF_INTERCEPT;
  else
    result = (voltage / RFMETER_DEF_SLOPE) + RFMETER_DEF_INTERCEPT;
 
  return result;
}
 
// --------------------- Arduino Setup ---------------------
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("---- Application Init ----");
  rfmeter_init();
}
 
// --------------------- Arduino Loop ----------------------
void loop() {
  uint16_t raw = rfmeter_read_raw();       // Read raw ADC value
  float voltage = (raw * RFMETER_DEF_VREF) / RFMETER_ADC_RESOLUTION; // Compute voltage
  float signal = rfmeter_get_signal_strength(voltage);
 
  Serial.print("Raw: ");
  Serial.print(raw);
  Serial.print(" | Voltage: ");
  Serial.print(voltage, 4);
  Serial.print(" V | Signal: ");
  Serial.print(signal, 2);
  Serial.println(" dBm");
 
  delay(1000);
}