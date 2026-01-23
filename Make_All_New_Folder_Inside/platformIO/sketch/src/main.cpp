#include <stdint.h>
#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include "Adafruit_TCS34725.h"
#include <ArduinoFFT.h>

// ----------------------------------------------------------------------------
// code for target board : Arduino Zero
// project : BiDiLTE test bench
// last change : Jan 23,2026
//
// Note: case 33 for mesuaring frequency at analog A2 only work in range of
// 100 to 500 hz because of alrady define sampleing rate.
// ----------------------------------------------------------------------------

/* -------------------------------------------------------------------------- */
/*                               Global Objects                                */
/* -------------------------------------------------------------------------- */

/* Global color sensor object (I2C) */
Adafruit_TCS34725 tcs(TCS34725_INTEGRATIONTIME_154MS, TCS34725_GAIN_4X);

/* -------------------------------------------------------------------------- */
/*                              Hardware Mapping                               */
/* -------------------------------------------------------------------------- */

typedef struct {
  const int WakeUp_Out;   // pin 2 : input from sensor - wakeup line
  const int Relay1;       // pin 3 : control of relay1 for main power supply ON OFF with 3.3v from PPK2
  const int Relay5;       // pin 4 : V plus control pin
  const int Relay2;       // pin 5 : control of relay2 for power supply depletion with 1kohm
  const int Relay3;       // pin 6 : control of relay3 for input of sensor - RX line
  const int Relay4;       // pin 7 : control of relay4 for output from sensor - TX line
  const int Magnet;       // pin 8 : output from sensor - busy line // magnet
  const int Alarm_In;     // pin 9 : output from sensor - alarm line
  const int GWBusy_Out;   // pin 10 : input from sensor - GW Busy line
} BoardPins_t;

// Board pin mapping (ordered by physical pin number for clarity and debugging)
static const BoardPins_t PINS = {
  2,   // WakeUp_Out
  3,   // Relay1
  4,   // Relay5
  5,   // Relay2
  6,   // Relay3
  7,   // Relay4
  8,   // Magnet
  9,   // Alarm_In
  10   // GWBusy_Out
};

/* -------------------------------------------------------------------------- */
/*                         Application Constants / Params                      */
/* -------------------------------------------------------------------------- */

/* USAGE OF OUTPUT PWM WITH THE BUZZER FUNCTION  */
const int speakerDuration = 4000;     // 4 seconds tone duration on pin 7
const int tone_freq300 = 300;         // frequency min of the pwm oscillation
const int tone_freq500 = 500;
const int tone_freq700 = 700;

/* AUDIO PARTS GENERATION : SINUS WAVE */
const uint16_t SAMPLECOUNT = 70;      // Number of samples to read in block
int *wavSamples;                      // aray to store sinewave points
// 600Hz = param sampleCount = 35 & sampleRate = 20000
// 300Hz = param sampleCount = 70 & sampleRate = 20000

/* -------------------------------------------------------------------------- */
/*                     Divider + Sampler Configuration (typedef)               */
/* -------------------------------------------------------------------------- */

typedef struct {
  // Divider settings
  const int DIVIDER_1;       // 1.5 volts
  const int DIVIDER_2;       // 0.903 volts
  const int DIVIDER_3;       // 0.445 volts
  const int DIVIDER_4;       // 0.251 volts
  const int DIVIDER_5;       // 0.155 volts
  const int DIVIDER_6;       // 0.095 volts
  const int DIVIDER_7;       // 0.072 volts
  const int DIVIDER_8;       // 0.015 volts
  const int DIVIDER_9;       // 0.009 volts

  // Tone sampler presets
  const uint32_t SAMPLER_1;  // 101 Hz
  const uint32_t SAMPLER_2;  // 202 Hz
  const uint32_t SAMPLER_3;  // 303 Hz
  const uint32_t SAMPLER_4;  // 404 Hz
  const uint32_t SAMPLER_5;  // 504 Hz
  const uint32_t SAMPLER_6;  // 605 Hz
  const uint32_t SAMPLER_7;  // 705 Hz
  const uint32_t SAMPLER_8;  // 805 Hz
  const uint32_t SAMPLER_9;  // 905 Hz
  const uint32_t SAMPLER_10; // 1005 Hz
  const uint32_t SAMPLER_11; // 1104 Hz
} AnalogToneConfig_t;

static const AnalogToneConfig_t TONE_CFG = {
  2,    // DIVIDER_1
  4,    // DIVIDER_2
  8,    // DIVIDER_3
  16,   // DIVIDER_4
  32,   // DIVIDER_5
  64,   // DIVIDER_6
  128,  // DIVIDER_7
  256,  // DIVIDER_8
  512,  // DIVIDER_9

  7000,   // SAMPLER_1
  14000,  // SAMPLER_2
  21000,  // SAMPLER_3
  28000,  // SAMPLER_4
  35000,  // SAMPLER_5
  42000,  // SAMPLER_6
  49000,  // SAMPLER_7
  56000,  // SAMPLER_8
  63000,  // SAMPLER_9
  70000,  // SAMPLER_10
  77000   // SAMPLER_11
};

/* -------------------------------------------------------------------------- */
/*                        Bitfield Meaning (readability)                       */
/* -------------------------------------------------------------------------- */

// InOutState bit layout (as used by your existing logic)
// Order from MSB to LSB: WakeUp_Out, Relay4, Relay3, GWBusy_Out, Relay2, Relay1
typedef enum {
  IO_WAKEUP_BIT = 0b100000,
  IO_RELAY4_BIT = 0b010000,
  IO_RELAY3_BIT = 0b001000,
  IO_GWBUSY_BIT = 0b000100,
  IO_RELAY2_BIT = 0b000010,
  IO_RELAY1_BIT = 0b000001
} IOStateMask_t;

static const unsigned int IO_DEFAULT_STATE = 0b011110;

/* -------------------------------------------------------------------------- */
/*                           Command Labels (enum)                             */
/* -------------------------------------------------------------------------- */

typedef enum {
  CMD_SLEEP_OFF   = 0,
  CMD_SLEEP_ON    = 1,
  CMD_FTDI_OFF    = 2,
  CMD_FTDI_ON     = 3,
  CMD_GWBUSY_OFF  = 4,
  CMD_GWBUSY_ON   = 5,
  CMD_MAGNET_OFF  = 6,
  CMD_ALARM_READ  = 7,
  CMD_DEP_ON      = 8,
  CMD_DEP_OFF     = 9,
  CMD_MAGNET_ON   = 10,

  CMD_POWER_ON    = 16,
  CMD_POWER_OFF   = 17,

  CMD_READ_STATE  = 20,
  CMD_SINUS_OFF   = 21,
  CMD_SINUS_ON    = 22,

  CMD_VPLUS_ON    = 23,
  CMD_VPLUS_OFF   = 24,

  CMD_VPLUS_READ  = 30,
  CMD_VMINUS_READ = 31,
  CMD_PEGEL_READ  = 32,
  CMD_FFT_READ    = 33,
  CMD_COLOR_DET   = 34,
  CMD_RGB_READ    = 35
} Command_t;

/* -------------------------------------------------------------------------- */
/*                       Runtime State (moved into a struct)                   */
/* -------------------------------------------------------------------------- */

typedef struct {
  unsigned int InOutState;

  // UART RS232 COMMUNICATION VARIABLES
  unsigned char receivedByte;   // received command byte from computer over uart rs232
  unsigned char subCommand;     // decoded received command from computer
  unsigned char value;          // digital input read buffer

  // AUDIO PARTS GENERATION : SINUS WAVE
  volatile uint16_t sIndex;     // Tracks sinewave points in array
  uint32_t sampleRate;          // sample rate of the sine wave
  unsigned char AnalogueOnOff;  // use to start/stop the analogue output
} Runtime_t;

static Runtime_t RT = {
  0,        // InOutState
  0, 0, 0,  // receivedByte, subCommand, value
  0,        // sIndex
  20000,    // sampleRate
  0         // AnalogueOnOff
};

/* -------------------------------------------------------------------------- */
/*                              FFT Configuration                              */
/* -------------------------------------------------------------------------- */

// Tunable params for measured frequency (Hz):
const uint32_t FFT_FS = 10000UL;      // sample rate for capture (10 kS/s)
const uint16_t FFT_N  = 2048;         // number of samples to capture (power of 2 recommended)
static float vReal[FFT_N];
static float vImag[FFT_N];

ArduinoFFT<float> FFT(vReal, vImag, FFT_N, FFT_FS);

/* -------------------------------------------------------------------------- */
/*                      Function prototypes (forward declarations)             */
/* -------------------------------------------------------------------------- */

// Sine generation and timer
void genSin(int sCount, int div);
void tcConfigure(uint32_t sampleRate);
bool tcIsSyncing();
void tcStartCounter();
void tcReset();
void tcDisable();
void TC5_Handler(void);

// ADC helpers
float readVoltage(uint8_t pin);
int readADCAverage(uint8_t pin);
float readVminus();
float readVplus();
int readADC_PEGEL();
float getFrequencyHz();

// Color helpers
void readColor(float &r, float &g, float &b);
bool isRed();
bool isGreen();
bool isBlue();

/* -------------------------------------------------------------------------- */
/*                           Professional Pin Init (function)                  */
/* -------------------------------------------------------------------------- */

static void initPinsAndDefaults()
{
  pinMode(PINS.WakeUp_Out, OUTPUT);        // use for power on/off the target board  (commands 0,1)
  digitalWrite(PINS.WakeUp_Out, LOW);      // startup with target board OFF

  pinMode(PINS.Relay4, OUTPUT);            // use for FTDI separation of TX line (commands 2,3)
  digitalWrite(PINS.Relay4, LOW);          // startup with line open

  pinMode(PINS.Relay3, OUTPUT);            // use for FTDI separation of RX line (commands 2,3)
  digitalWrite(PINS.Relay3, LOW);          // startup with line open

  pinMode(PINS.Relay2, OUTPUT);            // use for vcc capacitor depletion (commands 8,9)
  digitalWrite(PINS.Relay2, HIGH);         // startup with line open (no depletion)

  pinMode(PINS.Relay1, OUTPUT);            // use for vcc main line power on/off (command 16,17)
  digitalWrite(PINS.Relay1, LOW);          // startup with line open (vcc not provided to target PCBA)

  pinMode(PINS.Relay5, OUTPUT);            // V_plus control (commands 23,24)
  digitalWrite(PINS.Relay5, LOW);          // V_plus default state

  pinMode(PINS.Magnet, OUTPUT);            // input reading from sensor into arduino (commands 6)

  pinMode(PINS.Alarm_In, INPUT);           // input reading from sensor into arduino (commands 7)

  pinMode(PINS.GWBusy_Out, OUTPUT);        // use for signaling GW busy to sensor  (commands 4,5)
  digitalWrite(PINS.GWBusy_Out, LOW);      // startup with line OFF

  // pinMode(speakerOut, OUTPUT);           // setup pin for the buzzer (command 10,11,12,13)

  RT.InOutState = IO_DEFAULT_STATE;        // init global variable as the default digitalWrite settings for each port

  // Make sure these analog inputs are configured if you use them
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);
  pinMode(A4, INPUT);
  pinMode(A5, INPUT);
}

/* -------------------------------------------------------------------------- */
/*                                    Setup                                   */
/* -------------------------------------------------------------------------- */

void setup() {
  initPinsAndDefaults();

  Serial.begin(115200);               // setup UART commmunication to the computer (python program for commands)

  analogWriteResolution(10);          // set the Arduino DAC for 10 bits of resolution (max)
  analogReadResolution(12);           // set the Arduino ADC for 12 bits of resolution (max)

  wavSamples = (int *)malloc(SAMPLECOUNT * sizeof(int));  // Allocate the buffer where the samples are stored
  genSin(SAMPLECOUNT, TONE_CFG.DIVIDER_1);                // function generates sine wave
  RT.sIndex = 0;                                          // Set to zero to start from beginning of waveform
  // tcConfigure(RT.sampleRate); //setup the timer counter based off of the user entered sample rate

  // Initialize the TCS34725 color sensor (I2C on SDA/SCL pins)
  if (!tcs.begin()) {
    Serial.println("No TCS34725 found ... check your connections.");
    // while (1) { delay(10); }   // halt if sensor is missing
  }
}

/* -------------------------------------------------------------------------- */
/*                                     Loop                                   */
/* -------------------------------------------------------------------------- */

void loop() {

  if (Serial.available() > 0) {
    RT.receivedByte = Serial.readString().toInt();
    Serial.print(String(RT.receivedByte) + ": ");      // always return received command from computer (help for debug)

    switch (RT.receivedByte)
    {
      case CMD_SLEEP_OFF:   // sleep OFF target board
        digitalWrite(PINS.WakeUp_Out, HIGH);
        Serial.println("Sleep OFF");
        RT.InOutState = RT.InOutState | IO_WAKEUP_BIT;
        break;

      case CMD_SLEEP_ON:   // sleep ON target board
        digitalWrite(PINS.WakeUp_Out, LOW);
        Serial.println("Sleep ON");
        RT.InOutState = RT.InOutState & (unsigned int)(~IO_WAKEUP_BIT);
        break;

      case CMD_FTDI_OFF:   // relay's OPENED OFF target board (opened)
        digitalWrite(PINS.Relay4, HIGH);
        digitalWrite(PINS.Relay3, HIGH);
        Serial.println("FTDI OFF");
        RT.InOutState = RT.InOutState | (IO_RELAY4_BIT | IO_RELAY3_BIT);
        break;

      case CMD_FTDI_ON:   // sleep CLOSED ON target board (close)
        digitalWrite(PINS.Relay4, LOW);
        digitalWrite(PINS.Relay3, LOW);
        Serial.println("FDTI ON");
        RT.InOutState = RT.InOutState & (unsigned int)(~(IO_RELAY4_BIT | IO_RELAY3_BIT));
        break;

      case CMD_GWBUSY_OFF:   // GW busy OFF target board
        digitalWrite(PINS.GWBusy_Out, LOW);
        Serial.println("GWBusy OFF");
        RT.InOutState = RT.InOutState & (unsigned int)(~IO_GWBUSY_BIT);
        break;

      case CMD_GWBUSY_ON:   // GW busy ON target board
        digitalWrite(PINS.GWBusy_Out, HIGH);
        Serial.println("GWBusy ON");
        RT.InOutState = RT.InOutState | IO_GWBUSY_BIT;
        break;

      case CMD_MAGNET_OFF:   // write
        digitalWrite(PINS.Magnet, LOW);
        Serial.println("Magnet OFF");
        break;

      case CMD_MAGNET_ON:   // write
        digitalWrite(PINS.Magnet, HIGH);
        Serial.println("Magnet ON");
        break;

      case CMD_ALARM_READ:   // read
        RT.value = digitalRead(PINS.Alarm_In);
        if (0 == RT.value)
          Serial.println("Alarm OFF");
        else
          Serial.println("Alarm ON");
        break;

      case CMD_DEP_ON:   // start deplete VCC
        digitalWrite(PINS.Relay2, HIGH);
        Serial.println("Depletion ON");
        RT.InOutState = RT.InOutState & 0b111101;
        break;

      case CMD_DEP_OFF:   // stop deplete VCC
        digitalWrite(PINS.Relay2, LOW);
        Serial.println("Depletion OFF");
        RT.InOutState = RT.InOutState | 0b000010;
        break;

      case 0x80 ... 0xFF: // stop sinus output
        RT.subCommand = RT.receivedByte & 0x70;
        switch (RT.subCommand)
        {
          case 0x00:    //divider = 1
            genSin(SAMPLECOUNT, TONE_CFG.DIVIDER_1);
            Serial.print("Div 1 ");
            break;
          case 0x10:    //divider = 2
            genSin(SAMPLECOUNT, TONE_CFG.DIVIDER_2);
            Serial.print("Div 2 ");
            break;
          case 0x20:
            genSin(SAMPLECOUNT, TONE_CFG.DIVIDER_3);
            Serial.print("Div 3 ");
            break;
          case 0x30:
            genSin(SAMPLECOUNT, TONE_CFG.DIVIDER_4);
            Serial.print("Div 4 ");
            break;
          case 0x40:
            genSin(SAMPLECOUNT, TONE_CFG.DIVIDER_5);
            Serial.print("Div 5 ");
            break;
          case 0x50:
            genSin(SAMPLECOUNT, TONE_CFG.DIVIDER_6);
            Serial.print("Div 6 ");
            break;
          case 0x60:    //divider = 4
            genSin(SAMPLECOUNT, TONE_CFG.DIVIDER_7);
            Serial.print("Div 7 ");
            break;
          case 0x70:    //divider = 4
            genSin(SAMPLECOUNT, TONE_CFG.DIVIDER_8);
            Serial.print("Div 8 ");
            break;
        }

        RT.subCommand = RT.receivedByte & 0x0F;
        switch (RT.subCommand)
        {
          case 0x00:    // no Freq
            RT.AnalogueOnOff = 0;
            Serial.println("Tone OFF");
            break;
          case 0x01:
            RT.AnalogueOnOff = 1;
            RT.sampleRate = TONE_CFG.SAMPLER_1;
            Serial.println("Tone 100");
            break;
          case 0x02:
            RT.AnalogueOnOff = 1;
            RT.sampleRate = TONE_CFG.SAMPLER_2;
            Serial.println("Tone 200");
            break;
          case 0x03:
            RT.AnalogueOnOff = 1;
            RT.sampleRate = TONE_CFG.SAMPLER_3;
            Serial.println("Tone 300");
            break;
          case 0x04:
            RT.AnalogueOnOff = 1;
            RT.sampleRate = TONE_CFG.SAMPLER_4;
            Serial.println("Tone 400");
            break;
          case 0x05:
            RT.AnalogueOnOff = 1;
            RT.sampleRate = TONE_CFG.SAMPLER_5;
            Serial.println("Tone 500");
            break;
          case 0x06:
            RT.AnalogueOnOff = 1;
            RT.sampleRate = TONE_CFG.SAMPLER_6;
            Serial.println("Tone 600");
            break;
          case 0x07:
            RT.AnalogueOnOff = 1;
            RT.sampleRate = TONE_CFG.SAMPLER_7;
            Serial.println("Tone 700");
            break;
          case 0x08:
            RT.AnalogueOnOff = 1;
            RT.sampleRate = TONE_CFG.SAMPLER_8;
            Serial.println("Tone 800");
            break;
          case 0x09:
            RT.AnalogueOnOff = 1;
            RT.sampleRate = TONE_CFG.SAMPLER_9;
            Serial.println("Tone 900");
            break;
          case 0x0A:
            RT.AnalogueOnOff = 1;
            RT.sampleRate = TONE_CFG.SAMPLER_10;
            Serial.println("Tone 1000");
            break;
          case 0x0B:
            RT.AnalogueOnOff = 1;
            RT.sampleRate = TONE_CFG.SAMPLER_11;
            Serial.println("Tone 1100");
            break;
        }
        break;

      case CMD_POWER_ON:    // VCC close to target PCBA
        digitalWrite(PINS.Relay1, HIGH);
        Serial.println("Power ON");
        RT.InOutState = RT.InOutState & 0b111110;
        break;

      case CMD_POWER_OFF:    // VCC opened to target PCBA
        digitalWrite(PINS.Relay1, LOW);
        Serial.println("Power OFF");
        RT.InOutState = RT.InOutState | 0b000001;
        break;

      case CMD_READ_STATE:   // read back the current state of the outputs
        Serial.println(RT.InOutState); // in the following order from MSB to LSB
        break;

      case CMD_SINUS_OFF:
        tcDisable();            // stop the interrupt for sinus output on port A0
        analogWrite(A0, 0);     // force to have voltage output on A0 to 0 volt
        Serial.println("Sinus OFF");
        break;

      case CMD_SINUS_ON:
        tcConfigure(RT.sampleRate);    // restart the sinus output on port A0
        Serial.println("Sinus ON");
        break;

      case CMD_VPLUS_ON:    // VCC opened to target PCBA
        digitalWrite(PINS.Relay5, HIGH);
        Serial.println("V_Plus_On");
        // InOutState = InOutState | 0b000001; TO IMPLEMENT
        break;

      case CMD_VPLUS_OFF:    // VCC opened to target PCBA
        digitalWrite(PINS.Relay5, LOW);
        Serial.println("V_Plus_Off");
        // InOutState = InOutState | 0b000001; TO IMPLEMENT
        break;

      case CMD_VPLUS_READ:    // Read Vplus (ADC on A5)
        Serial.print("Vplus ADC = ");
        Serial.println(readVplus());
        break;

      case CMD_VMINUS_READ:   // Read Vminus (ADC on A3)
        Serial.print("Vminus ADC = ");
        Serial.println(readVminus());
        break;

      case CMD_PEGEL_READ:    // Read PEGEL (ADC on A1)
        Serial.print("PEGEL ADC = ");
        Serial.println(readADC_PEGEL());
        break;

      case CMD_FFT_READ:      // Read FFT (ADC on A2)
        Serial.print("freq = ");
        Serial.println(getFrequencyHz());
        break;

      case CMD_COLOR_DET:     // Color detection
        if (isRed()) {
          Serial.println("Detected: RED");
        } else if (isGreen()) {
          Serial.println("Detected: GREEN");
        } else if (isBlue()) {
          Serial.println("Detected: BLUE");
        } else {
          Serial.println("Detected: UNKNOWN");
        }
        break;

      case CMD_RGB_READ:      // Read full normalized RGB values
      {
        float r, g, b;
        readColor(r, g, b);
        Serial.print("RGB(norm 0-256): ");
        Serial.print(r); Serial.print(", ");
        Serial.print(g); Serial.print(", ");
        Serial.println(b);
      }
      break;

      default:                // always return unknowed command from computer (help for debug)
        Serial.println("Unknown");
        break;
    }
  }
}

/* -------------------------------------------------------------------------- */
/*                          Sine Wave + Timer Functions                        */
/* -------------------------------------------------------------------------- */

// This function generates a sine wave and stores it in the wavSamples array
// The input argument is the number of points the sine wave is made up of
void genSin(int sCount, int div) {
  const float pi2 = 6.283; //2 x pi
  float in;

  for (int i = 0; i < sCount; i++)
  { // loop to build sine wave based on sample count
    in = pi2 * (1 / (float)sCount) * (float)i;               // calculate value in radians for sin()
    // HERE it is possible to divide the amplitude of the waveform   /1 /2 /4
    wavSamples[i] = ((int)(sin(in) * 511.5 + 511.5) / div);  // Calculate sine wave value and offset based on DAC resolution 511.5 = 1023/2
  }
}

// Configures the TC to generate output events at the sample frequency.
// Configures the TC in Frequency Generation mode, with an event output once
// each time the audio sample frequency period expires.
void tcConfigure(uint32_t sampleRate)
{
  // Enable GCLK for TCC2 and TC5 (timer counter input clock)
  GCLK->CLKCTRL.reg = (uint16_t)(GCLK_CLKCTRL_CLKEN | GCLK_CLKCTRL_GEN_GCLK0 | GCLK_CLKCTRL_ID(GCM_TC4_TC5));
  while (GCLK->STATUS.bit.SYNCBUSY);

  tcReset(); //reset TC5

  TC5->COUNT16.CTRLA.reg |= TC_CTRLA_MODE_COUNT16;         // Set Timer counter Mode to 16 bits
  TC5->COUNT16.CTRLA.reg |= TC_CTRLA_WAVEGEN_MFRQ;         // Set TC5 mode as match frequency
  TC5->COUNT16.CTRLA.reg |= TC_CTRLA_PRESCALER_DIV1 | TC_CTRLA_ENABLE;   // Set prescaler and enable TC5

  // Set TC5 timer counter based off of the system clock and the user defined sample rate or waveform
  TC5->COUNT16.CC[0].reg = (uint16_t)(SystemCoreClock / sampleRate - 1);
  while (tcIsSyncing());

  NVIC_DisableIRQ(TC5_IRQn);           // Configure interrupt request
  NVIC_ClearPendingIRQ(TC5_IRQn);
  NVIC_SetPriority(TC5_IRQn, 0);
  NVIC_EnableIRQ(TC5_IRQn);

  TC5->COUNT16.INTENSET.bit.MC0 = 1;   // Enable the TC5 interrupt request
  while (tcIsSyncing());               // Wait until TC5 is done syncing
}

bool tcIsSyncing()
{ // Function that is used to check if TC5 is done syncing; returns true when it is done syncing
  return TC5->COUNT16.STATUS.reg & TC_STATUS_SYNCBUSY;
}

void tcStartCounter()
{ // This function enables TC5 and waits for it to be ready
  TC5->COUNT16.CTRLA.reg |= TC_CTRLA_ENABLE;      // Set the CTRLA register
  while (tcIsSyncing());                          // Wait until snyc'd
}

void tcReset()
{ // Reset TC5
  TC5->COUNT16.CTRLA.reg = TC_CTRLA_SWRST;
  while (tcIsSyncing());
  while (TC5->COUNT16.CTRLA.bit.SWRST);
}

void tcDisable()
{ // Disable TC5
  TC5->COUNT16.CTRLA.reg &= ~TC_CTRLA_ENABLE;
  while (tcIsSyncing());
}

void TC5_Handler(void)
{
  analogWrite(A0, wavSamples[RT.sIndex]);
  TC5->COUNT16.INTFLAG.bit.MC0 = 1;

  if (++RT.sIndex >= SAMPLECOUNT)
  {
    RT.sIndex = 0;
    tcDisable();               // disable and reset timer counter
    tcReset();
    tcConfigure(RT.sampleRate); // setup the timer counter based off of the user entered sample rate
  }

  tcStartCounter();            // start timer, once timer is done interrupt will occur and DAC value will be updated
}

/* -------------------------------------------------------------------------- */
/*                                  ADC Helpers                                */
/* -------------------------------------------------------------------------- */

float readVoltage(uint8_t pin) {
  int raw = readADCAverage(pin);
  float voltage = (raw * 3.3) / 4095.0;   // if 12-bit resolution
  return voltage;
}

// --------- ADC helpers ---------
int readADCAverage(uint8_t pin) {
  long sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += analogRead(pin);
  }
  // SerialUSB.println((int)(sum / 10));
  return (int)(sum / 10);
}

// Named channels (change pins to match your wiring as needed)
float readVminus()     { return readVoltage(A3); }
float readVplus()      { return readVoltage(A5); }
int readADC_PEGEL()    { return readADCAverage(A1); }

// -------------------- Frequency-capture helper --------------------
// Call readADC_FFT() to block, capture and return measured frequency (Hz).
inline bool micros_less(uint32_t a, uint32_t b) {
  // return true if a < b considering wrap-around
  return (int32_t)(a - b) < 0;
}

float getFrequencyHz()
{
  const uint32_t period_us = 1000000UL / FFT_FS;
  uint32_t t0 = micros();

  // ---- Sample ADC ----
  for (uint16_t i = 0; i < FFT_N; i++) {
    while ((int32_t)(micros() - (t0 + i * period_us)) < 0);
    vReal[i] = (float)analogRead(A4);
    vImag[i] = 0.0f;
  }

  // ---- Remove DC ----
  float mean = 0.0f;
  for (uint16_t i = 0; i < FFT_N; i++) mean += vReal[i];
  mean /= FFT_N;
  for (uint16_t i = 0; i < FFT_N; i++) vReal[i] -= mean;

  // ---- FFT ----
  FFT.windowing(FFTWindow::Hann, FFTDirection::Forward);
  FFT.compute(FFTDirection::Forward);
  FFT.complexToMagnitude();

  // ---- Find peak bin ----
  uint16_t peakBin = 1;
  float peakMag = vReal[1];

  for (uint16_t i = 2; i < FFT_N / 2; i++) {
    if (vReal[i] > peakMag) {
      peakMag = vReal[i];
      peakBin = i;
    }
  }

  // ---- Convert to frequency (INT) ----
  float frequencyHz = ((float)peakBin * FFT_FS / FFT_N) / 8.69;
  return frequencyHz;
}

// -------------------- end frequency helper --------------------

/* -------------------------------------------------------------------------- */
/*                                 Color Helpers                               */
/* -------------------------------------------------------------------------- */

void readColor(float &r, float &g, float &b) {
  uint16_t clear, red, green, blue;
  tcs.getRawData(&red, &green, &blue, &clear);

  uint32_t sum = clear;
  if (sum == 0) sum = 1;     // prevent div/0

  r = (float)red   / sum * 256.0f;
  g = (float)green / sum * 256.0f;
  b = (float)blue  / sum * 256.0f;
}

bool isRed()   { float r, g, b; readColor(r, g, b); return (r > g && r > b); }
bool isGreen() { float r, g, b; readColor(r, g, b); return (g > r && g > b); }
bool isBlue()  { float r, g, b; readColor(r, g, b); return (b > r && b > g); }

// End of file
