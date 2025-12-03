#include <SPI.h>

// ----------------- RF Meter Click config -----------------
const int PIN_CS      = 53;     // CS from Click -> Mega 53 (or change to your pin)
const float RFMETER_VREF       = 2.5f;    // AP7331 reference on the Click
const float RFMETER_ADC_RES    = 4096.0f; // 12-bit ADC
const uint16_t RFMETER_FILTER_USEFULL_DATA = 0x1FFF; // keep 13 LSBs, then >>1

// You must calibrate these for your setup (frequency, etc.)
const float RFMETER_DEF_SLOPE     = -0.025f;   // V/dB (~22mV/dB, positive)
const float RFMETER_DEF_INTERCEPT = 20.0f;   // example offset in dBm

// Limits used in original driver (approx, tweak if needed)
const float RFMETER_DEF_LIMIT_HIGH = 2.0f;    // V
const float RFMETER_DEF_LIMIT_LOW  = 0.5f;    // V

SPISettings rfSpiSettings(250000, MSBFIRST, SPI_MODE0);

// ----------------- Low-level read (Arduino version) -----------------
uint16_t rfmeter_read_data()
{
    uint8_t rx_buf[2];

    SPI.beginTransaction(rfSpiSettings);
    digitalWrite(PIN_CS, LOW);

    // Read 2 bytes by clocking out dummy 0x00
    rx_buf[0] = SPI.transfer(0x00);
    rx_buf[1] = SPI.transfer(0x00);

    digitalWrite(PIN_CS, HIGH);
    SPI.endTransaction();

    uint16_t result = ((uint16_t)rx_buf[0] << 8) | rx_buf[1];
    return result;
}

// ----------------- Same functions as MikroE driver -----------------
uint16_t rfmeter_get_raw_data()
{
    uint16_t result = rfmeter_read_data();
    result &= RFMETER_FILTER_USEFULL_DATA; // 0x1FFF
    result >>= 1;                          // align 12 bits (0..4095)
    return result;
}

float rfmeter_get_voltage()
{
    uint16_t reading = rfmeter_get_raw_data();
    float v = (float)reading;
    v *= RFMETER_VREF;
    v /= RFMETER_ADC_RES;   // /4095
    return v;
}

float rfmeter_get_signal_strength(float slope, float intercept)
{
    float voltage = rfmeter_get_voltage();
    float result;

    if (voltage > RFMETER_DEF_LIMIT_HIGH)
    {
        result = (RFMETER_DEF_LIMIT_HIGH / slope) + intercept;
    }
    else if (voltage < RFMETER_DEF_LIMIT_LOW)
    {
        result = (RFMETER_DEF_LIMIT_LOW / slope) + intercept;
    }
    else
    {
        result = (voltage / slope) + intercept;
    }

    return result;
}

// ----------------- Example usage on Arduino Mega -----------------
void setup()
{
    Serial.begin(115200);
    while (!Serial) {}

    pinMode(PIN_CS, OUTPUT);
    digitalWrite(PIN_CS, HIGH);

    SPI.begin();  // uses 50=MISO, 51=MOSI, 52=SCK, 53=SS on Mega

    Serial.println("RF Meter Click (Arduino/Mega) test");
}

void loop()
{
    uint16_t raw     = rfmeter_get_raw_data();
    float vout       = rfmeter_get_voltage();
    float signal_dbm = rfmeter_get_signal_strength(RFMETER_DEF_SLOPE,
                                                   RFMETER_DEF_INTERCEPT);

    Serial.print("Raw: ");
    Serial.print(raw);
    Serial.print("  Vout: ");
    Serial.print(vout, 4);
    Serial.print(" V  ~P: ");
    Serial.print(signal_dbm, 1);
    Serial.println(" dBm (approx)");

    delay(500);
}
