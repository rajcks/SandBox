// language reference link : https://www.arduino.cc/reference/en/
#include <stdint.h>
#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include "Adafruit_TCS34725.h"

// code for target board : Arduino Zero
// project : BiDiLTE test bench
// last change : Nov 11,2025

/*
Note: case 33 for mesuaring frequency at analog A2 only work in range of 100 to 500 hz because of alrady define sampleing rate.
*/

/* Global color sensor object (I2C) */
Adafruit_TCS34725 tcs(TCS34725_INTEGRATIONTIME_154MS, TCS34725_GAIN_4X);

/* USAGE OF OUTPUT PWM WITH THE BUZZER FUNCTION  */
const int speakerDuration = 4000;     // 4 seconds tone duration on pin 7
const int tone_freq300 = 300;         // frequency min of the pwm oscillation
const int tone_freq500 = 500;
const int tone_freq700 = 700;

/* PORTS CONFIGURATION ON THE TARGET BOARD */
const int WakeUp_Out = 2;             // pin 2 : input from sensor - wakeup line
const int Relay4 = 7;                 // pin 3 : control of relay4 for output from sensor - TX line
const int Relay3 = 6;                 // pin 4 : control of relay3 for input of sensor - RX line
const int Relay2 = 5;                 // pin 5 : control of relay2 for power supply depletion with 1kohm
const int Relay1 = 3;                 // pin 6 : control of relay1 for main power supply ON OFF with 3.3v from PPK2
const int speakerOut = 4;             // pin 8  : output pin to speaker to generate audio tone
const int Busy_In = 8;                // pin 8 : output from sensor - busy line
const int Alarm_In = 9;               // pin 9 : output from sensor - alarm line
const int GWBusy_Out = 10;            // pin 10 : input from sensor - GW Busy line
unsigned int InOutState = 0;

/* UART RS232 COMMUNICATION VARIABLES */
unsigned char receivedByte;           // received command byte from computer over uart rs232
unsigned char subCommand;             // decoded received command from computer
unsigned char value;                  // digital input read buffer

/* AUDIO PARTS GENERATION : SINUS WAVE */
volatile uint16_t sIndex;                  //Tracks sinewave points in array
const uint16_t SAMPLECOUNT = 70;  // Number of samples to read in block
int *wavSamples;                      //aray to store sinewave points
uint32_t sampleRate = 20000;          //sample rate of the sine wave
// 600Hz = param sampleCount = 35 & sampleRate = 20000
// 300Hz = param sampleCount = 70 & sampleRate = 20000
unsigned char AnalogueOnOff = 0;      // use to start/stop the analogue output

const int DIVIDER_1 = 2;      // 1.5 volts
const int DIVIDER_2 = 4;      // 0.903 volts
const int DIVIDER_3 = 8;      // 0.445 volts
const int DIVIDER_4 = 16;      // 0.251 volts
const int DIVIDER_5 = 32;      // 0.155 volts
const int DIVIDER_6 = 64;      // 0.095 volts
const int DIVIDER_7 = 128;      // 0.072 volts
const int DIVIDER_8 = 256;      // 0.015 volts
const int DIVIDER_9 = 512;      // 0.009 volts

const uint32_t SAMPLER_1 = 7000;   // 101 Hz
const uint32_t SAMPLER_2 = 14000;   // 202 Hz
const uint32_t SAMPLER_3 = 21000;   // 303 Hz
const uint32_t SAMPLER_4 = 28000;   // 404 Hz
const uint32_t SAMPLER_5 = 35000;   // 504 Hz
const uint32_t SAMPLER_6 = 42000;   // 605 Hz
const uint32_t SAMPLER_7 = 49000;   // 705 Hz
const uint32_t SAMPLER_8 = 56000;   // 805 Hz
const uint32_t SAMPLER_9 = 63000;   // 905 Hz
const uint32_t SAMPLER_10 = 70000;   // 1005 Hz
const uint32_t SAMPLER_11 = 77000;   // 1104 Hz

// Tunable params for measured frequency (Hz):
const uint32_t FFT_FS = 10000UL;      // sample rate for capture (10 kS/s)
const uint16_t FFT_N = 2048;          // number of samples to capture (power of 2 recommended)

/* ---------- Function prototypes (forward declarations) ---------- */
void genSin(int sCount, int div);
void tcConfigure(uint32_t sampleRate);
bool tcIsSyncing();
void tcStartCounter();
void tcReset();
void tcDisable();
void TC5_Handler(void);
// ---- ADC helper prototypes ----
float readVoltage(uint8_t pin);
int readADCAverage(uint8_t pin);
float readVminus();
float readVplus();
int readADC_PEGEL();
float readADC_FFT();

// ---- Color helper prototypes ----
void readColor(float &r, float &g, float &b);
bool isRed();
bool isGreen();
bool isBlue();
/* ---------------------------------------------------------------- */

void setup() {
  // setup pins for power control of target board
  pinMode(WakeUp_Out, OUTPUT);        // use for power on/off the target board  (commands 0,1)
  digitalWrite(WakeUp_Out, LOW);      // startup with target board OFF
  pinMode(Relay4, OUTPUT);            // use for FTDI separation of TX line (commands 2,3)
  digitalWrite(Relay4, LOW);         // startup with line open
  pinMode(Relay3, OUTPUT);            // use for FTDI separation of RX line (commands 2,3)
  digitalWrite(Relay3, LOW);         // startup with line open
  pinMode(Relay2, OUTPUT);            // use for vcc capacitor depletion (commands 8,9)
  digitalWrite(Relay2, HIGH);         // startup with line open (no depletion)
  pinMode(Relay1, OUTPUT);            // use for vcc main line power on/off (command 16,17)
  digitalWrite(Relay1, LOW);         // startup with line open (vcc not provided to target PCBA)
  pinMode(Busy_In, INPUT);            // input reading from sensor into arduino (commands 6)
  pinMode(Alarm_In, INPUT);           // input reading from sensor into arduino (commands 7)
  pinMode(GWBusy_Out, OUTPUT);        // use for signaling GW busy to sensor  (commands 4,5)
  digitalWrite(GWBusy_Out, LOW);      // startup with line OFF
  pinMode(speakerOut, OUTPUT);        // setup pin for the buzzer (command 10,11,12,13)
  InOutState = 0b011110;              // init global variable as the default digitalWrite settings for each port

  // Make sure these analog inputs are configured if you use them
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);
  pinMode(A4, INPUT);

  Serial.begin(115200);               // setup UART commmunication to the computer (python program for commands)

  analogWriteResolution(10);          //set the Arduino DAC for 10 bits of resolution (max)
  analogReadResolution(12);           //set the Arduino ADC for 12 bits of resolution (max)
  wavSamples = (int *) malloc(SAMPLECOUNT * sizeof(int));     // Allocate the buffer where the samples are stored
  genSin(SAMPLECOUNT, DIVIDER_1);                //function generates sine wave
  sIndex = 0;   //Set to zero to start from beginning of waveform
  // tcConfigure(sampleRate); //setup the timer counter based off of the user entered sample rate

  // Initialize the TCS34725 color sensor (I2C on SDA/SCL pins)
  if (!tcs.begin()) {
    Serial.println("No TCS34725 found ... check your connections.");
    // while (1) { delay(10); }   // halt if sensor is missing
  }
}

void loop() {

    if (Serial.available() > 0) {
    receivedByte = Serial.readString().toInt();
    Serial.print(String(receivedByte) + ": ");      // always return received command from computer (help for debug)
    switch(receivedByte)
    { case 0:   // sleep OFF target board
        digitalWrite(WakeUp_Out, HIGH);
        Serial.println("Sleep OFF");
        InOutState = InOutState | 0b100000;
        break;
      case 1:   // sleep ON target board
        digitalWrite(WakeUp_Out, LOW);
        Serial.println("Sleep ON");
        InOutState = InOutState & 0b011111;
        break;
      case 2:   // relay's OPENED OFF target board (opened)
        digitalWrite(Relay4, HIGH);
        digitalWrite(Relay3, HIGH);
        Serial.println("FTDI OFF");
        InOutState = InOutState | 0b011000;
        break;
      case 3:   // sleep CLOSED ON target board (close)
        digitalWrite(Relay4, LOW);
        digitalWrite(Relay3, LOW);
        Serial.println("FDTI ON");
        InOutState = InOutState & 0b100111;
        break;
      case 4:   // GW busy OFF target board
        digitalWrite(GWBusy_Out, LOW);
        Serial.println("GWBusy OFF");
        InOutState = InOutState & 0b111011;
        break;
      case 5:   // GW busy ON target board
        digitalWrite(GWBusy_Out, HIGH);
        Serial.println("GWBusy ON");
        InOutState = InOutState | 0b000100;
        break;
      case 6:   // read
        value = digitalRead(Busy_In);
        if (0 == value)
          Serial.println("Busy OFF");
        else
          Serial.println("Busy ON");
        break;
      case 7:   // read
        value = digitalRead(Alarm_In);
        if (0 == value)
          Serial.println("Alarm OFF");
        else
          Serial.println("Alarm ON");
        break;
      case 8:   // start deplete VCC
        digitalWrite(Relay2, HIGH);
        Serial.println("Depletion ON");
        InOutState = InOutState & 0b111101;
        break;
      case 9:   // stop deplete VCC
        digitalWrite(Relay2, LOW);
        Serial.println("Depletion OFF");
        InOutState = InOutState | 0b000010;
        break;
      case 0x80 ... 0xFF: // stop sinus output
        subCommand = receivedByte & 0x70;
        switch(subCommand)
        { case 0x00:    //divider = 1
            genSin(SAMPLECOUNT, DIVIDER_1);
            Serial.print("Div 1 ");
            break;
          case 0x10:    //divider = 2
            genSin(SAMPLECOUNT, DIVIDER_2);
            Serial.print("Div 2 ");
            break;
          case 0x20:
            genSin(SAMPLECOUNT, DIVIDER_3);
            Serial.print("Div 3 ");
            break;
          case 0x30:
            genSin(SAMPLECOUNT, DIVIDER_4);
            Serial.print("Div 4 ");
            break;
          case 0x40:
            genSin(SAMPLECOUNT, DIVIDER_5);
            Serial.print("Div 5 ");
            break;
          case 0x50:
            genSin(SAMPLECOUNT, DIVIDER_6);
            Serial.print("Div 6 ");
            break;
          case 0x60:    //divider = 4
            genSin(SAMPLECOUNT, DIVIDER_7);
            Serial.print("Div 7 ");
            break;
          case 0x70:    //divider = 4
            genSin(SAMPLECOUNT, DIVIDER_8);
            Serial.print("Div 8 ");
            break;
        }
        subCommand = receivedByte & 0x0F;
        switch(subCommand)
        { case 0x00:    // no Freq
            AnalogueOnOff = 0;
            Serial.println("Tone OFF");
            break;
          case 0x01:
            AnalogueOnOff = 1;
            sampleRate = SAMPLER_1;
            Serial.println("Tone 100");
            break;
          case 0x02:
            AnalogueOnOff = 1;
            sampleRate = SAMPLER_2;
            Serial.println("Tone 200");
            break;
          case 0x03:
            AnalogueOnOff = 1;
            sampleRate = SAMPLER_3;
            Serial.println("Tone 300");
            break;
          case 0x04:
            AnalogueOnOff = 1;
            sampleRate = SAMPLER_4;
            Serial.println("Tone 400");
            break;
          case 0x05:
            AnalogueOnOff = 1;
            sampleRate = SAMPLER_5;
            Serial.println("Tone 500");
            break;
          case 0x06:
            AnalogueOnOff = 1;
            sampleRate = SAMPLER_6;
            Serial.println("Tone 600");
            break;
          case 0x07:
            AnalogueOnOff = 1;
            sampleRate = SAMPLER_7;
            Serial.println("Tone 700");
            break;
          case 0x08:
            AnalogueOnOff = 1;
            sampleRate = SAMPLER_8;
            Serial.println("Tone 800");
            break;
          case 0x09:
            AnalogueOnOff = 1;
            sampleRate = SAMPLER_9;
            Serial.println("Tone 900");
            break;
          case 0x0A:
            AnalogueOnOff = 1;
            sampleRate = SAMPLER_10;
            Serial.println("Tone 1000");
            break;
          case 0x0B:
            AnalogueOnOff = 1;
            sampleRate = SAMPLER_11;
            Serial.println("Tone 1100");
            break;
        }
        break;
      // case 12:    // sound 300Hz
      //   tone(speakerOut, tone_freq300, speakerDuration);  // 300Hz for 4 sec on pin 7
      //   Serial.println("Sound 300");
      //   break;
      // case 13:    // sound 500Hz
      //   tone(speakerOut, tone_freq500, speakerDuration);  // 500Hz for 4 sec on pin 7
      //   Serial.println("Sound 500");
      //   break;
      // case 14:    // sound 700Hz
      //   tone(speakerOut, tone_freq700, speakerDuration);  // 700Hz for 4 sec on pin 7
      //   Serial.println("Sound 800");
      //   break;
      //case 15:    // NO sound
      //  noTone(speakerOut);
      //  Serial.println("Sound OFF");
      //  break;
      case 16:    // VCC close to target PCBA
        digitalWrite(Relay1, HIGH);
        Serial.println("Power ON");
        InOutState = InOutState & 0b111110;
        break;
      case 17:    // VCC opened to target PCBA
        digitalWrite(Relay1, LOW);
        Serial.println("Power OFF");
        InOutState = InOutState | 0b000001;
        break;
      case 20:    // read back the current state of the outputs : WakeUp_Out, Relay4, Relay3, Relay2, Relay1, GWBusy_Out
        Serial.println(InOutState); // in the following order from MSB to LSB
        break;
      case 21:
        tcDisable();            // stop the interrupt for sinus output on port A0
        analogWrite(A0, 0);     // force to have voltage output on A0 to 0 volt
        Serial.println("Sinus OFF");
        break;
      case 22:
        tcConfigure(sampleRate);      // restart the sinus output on port A0
        Serial.println("Sinus ON");
        break;
      case 30:    // Read Vplus (ADC on A4)
        Serial.print("Vplus ADC = ");
        Serial.println(readVplus());
        break;
      case 31:    // Read Vminus (ADC on A3)
        Serial.print("Vminus ADC = ");
        Serial.println(readVminus());
        break;
      case 32:    // Read PEGEL (ADC on A1)
        Serial.print("PEGEL ADC = ");
        Serial.println(readADC_PEGEL());
        break;
      case 33:    // Read FFT (ADC on A2)
        Serial.print("freq = ");
        Serial.println(readADC_FFT());
        break;
      case 34:    // Color detection
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
      case 35:    // Read full normalized RGB values
        {
          float r, g, b;
          readColor(r, g, b);
          Serial.print("RGB(norm 0-256): ");
          Serial.print(r); Serial.print(", ");
          Serial.print(g); Serial.print(", ");
          Serial.println(b);
        }
        break;
      default:        // always return unknowed command from computer (help for debug)
        Serial.println("Unknown");
        break;
    }
  }
}

// This function generates a sine wave and stores it in the wavSamples array
// The input argument is the number of points the sine wave is made up of
void genSin(int sCount, int div) {
 const float pi2 = 6.283; //2 x pi
 float in;

 for(int i=0; i<sCount;i++)
 { // loop to build sine wave based on sample count
   in = pi2*(1/(float)sCount)*(float)i;               // calculate value in radians for sin()
   // HERE it is possible to divide the amplitude of the waveform   /1 /2 /4
   wavSamples[i] = ((int)(sin(in)*511.5 + 511.5)/div);  // Calculate sine wave value and offset based on DAC resolution 511.5 = 1023/2
 }
}

// Configures the TC to generate output events at the sample frequency.
// Configures the TC in Frequency Generation mode, with an event output once
// each time the audio sample frequency period expires.
void tcConfigure(uint32_t sampleRate)
{
 // Enable GCLK for TCC2 and TC5 (timer counter input clock)
 GCLK->CLKCTRL.reg = (uint16_t) (GCLK_CLKCTRL_CLKEN | GCLK_CLKCTRL_GEN_GCLK0 | GCLK_CLKCTRL_ID(GCM_TC4_TC5)) ;
 while (GCLK->STATUS.bit.SYNCBUSY);
 tcReset(); //reset TC5
 TC5->COUNT16.CTRLA.reg |= TC_CTRLA_MODE_COUNT16;         // Set Timer counter Mode to 16 bits
 TC5->COUNT16.CTRLA.reg |= TC_CTRLA_WAVEGEN_MFRQ;         // Set TC5 mode as match frequency
 TC5->COUNT16.CTRLA.reg |= TC_CTRLA_PRESCALER_DIV1 | TC_CTRLA_ENABLE;   // Set prescaler and enable TC5
 // Set TC5 timer counter based off of the system clock and the user defined sample rate or waveform
 TC5->COUNT16.CC[0].reg = (uint16_t) (SystemCoreClock / sampleRate - 1);
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

void TC5_Handler (void)
{
  analogWrite(A0, wavSamples[sIndex]);
  TC5->COUNT16.INTFLAG.bit.MC0 = 1;
  if (++sIndex >= SAMPLECOUNT)
  { sIndex = 0;
    tcDisable();             // disable and reset timer counter
    tcReset();
    tcConfigure(sampleRate); // setup the timer counter based off of the user entered sample rate
  }
  tcStartCounter();          // start timer, once timer is done interrupt will occur and DAC value will be updated
}

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
int readADC_PEGEL()  { return readADCAverage(A1); }
// float readADC_FFT()    { return readVoltage(A2); }
// -------------------- Frequency-capture helper --------------------
// Call readADC_FFT() to block, capture and return measured frequency (Hz).
inline bool micros_less(uint32_t a, uint32_t b) {
  // return true if a < b considering wrap-around
  return (int32_t)(a - b) < 0;
}

float readADC_FFT() {
  const uint32_t targetFs = FFT_FS;
  const uint16_t N = FFT_N;
  if (N < 16) return -1.0f;

  // Allocate buffers on heap to avoid large .bss usage
  uint16_t *buf = (uint16_t*) malloc((size_t)N * sizeof(uint16_t));
  uint32_t *t_us = (uint32_t*)  malloc((size_t)N * sizeof(uint32_t));
  if (!buf || !t_us) {
    if (buf) free(buf);
    if (t_us) free(t_us);
    Serial.println("ERR: not enough RAM for capture; reduce FFT_N");
    return -1.0f;
  }

  const uint32_t period_us = (uint32_t)(1000000UL / targetFs);
  uint32_t t0 = micros();
  uint32_t deadline = t0 + (uint32_t)(N + 1000) * max(1u, period_us);

  for (uint32_t i = 0; i < N; ++i) {
    uint32_t target = t0 + i * period_us;
    while ((int32_t)(micros() - target) < 0) {
      if ((int32_t)(micros() - deadline) >= 0) { // timeout
        free(buf); free(t_us);
        return -1.0f;
      }
      // busy wait
    }
    uint32_t ts = micros();
    uint16_t s = analogRead(A2);
    buf[i] = s;
    t_us[i] = ts;
  }

  // actual sampling rate (optional debug)
  uint32_t total_us = (t_us[N-1] >= t_us[0]) ? (t_us[N-1] - t_us[0])
                    : (uint32_t)((uint64_t)t_us[N-1] + (uint64_t)(UINT32_MAX - t_us[0]) + 1ULL);
  float actualFs = 0.0f;
  if (total_us > 0) actualFs = (float)(N - 1) * 1e6f / (float)total_us;
  // Serial.print("actualFs="); Serial.println(actualFs);  // enable for debug

  // find midpoint
  uint16_t minv = 0xFFFF, maxv = 0;
  for (uint16_t i = 0; i < N; ++i) {
    uint16_t v = buf[i];
    if (v < minv) minv = v;
    if (v > maxv) maxv = v;
  }
  float mid = (minv + maxv) * 0.5f;

  // allocate crossings buffer (float to save RAM)
  uint16_t maxCross = N / 2;
  float *crossTimes = (float*) malloc((size_t)maxCross * sizeof(float));
  if (!crossTimes) {
    free(buf); free(t_us);
    Serial.println("ERR: not enough RAM for crossings; reduce FFT_N");
    return -1.0f;
  }

  // gather rising crossings using timestamp interpolation
  uint16_t crossCount = 0;
  double t0_s = (double)t_us[0] * 1e-6;
  for (uint16_t i = 1; i < N; ++i) {
    float s1 = (float)buf[i - 1];
    float s2 = (float)buf[i];
    if (s1 < mid && s2 >= mid) {
      float denom = (s2 - s1);
      float frac = 0.0f;
      if (fabs(denom) > 1e-6f) frac = (mid - s1) / denom;
      uint32_t t1 = t_us[i - 1];
      uint32_t t2 = t_us[i];
      uint32_t dt_us = (t2 >= t1) ? (t2 - t1) : (uint32_t)((uint64_t)t2 + (uint64_t)(UINT32_MAX - t1) + 1ULL);
      double t_cross_s = ((double)t1 + (double)frac * (double)dt_us) * 1e-6;
      float t_rel = (float)(t_cross_s - t0_s); // seconds relative to start
      if (crossCount < maxCross) crossTimes[crossCount++] = t_rel;
    }
  }

  float result = -1.0f;
  if (crossCount >= 2) {
    double sumPeriods = 0.0;
    int validPeriods = 0;
    const double MIN_PERIOD = 1.0 / 2000.0; // 0.5 ms
    const double MAX_PERIOD = 1.0;          // 1 s
    for (int k = 1; k < crossCount; ++k) {
      double p = (double)crossTimes[k] - (double)crossTimes[k - 1];
      if (p > MIN_PERIOD && p < MAX_PERIOD) {
        sumPeriods += p;
        ++validPeriods;
      }
    }
    if (validPeriods > 0) {
      double meanPeriod = sumPeriods / validPeriods;
      result = (float)(1.0 / meanPeriod);
    }
  }

  free(buf);
  free(t_us);
  free(crossTimes);
  return result;
}

// -------------------- end frequency helper --------------------

// --------- Color helpers ---------
void readColor(float &r, float &g, float &b) {
  uint16_t clear, red, green, blue;
  tcs.getRawData(&red, &green, &blue, &clear);

  uint32_t sum = clear;
  if (sum == 0) sum = 1;     // prevent div/0

  r = (float)red   / sum * 256.0f;
  g = (float)green / sum * 256.0f;
  b = (float)blue  / sum * 256.0f;
}

bool isRed()   { float r,g,b; readColor(r,g,b); return (r > g && r > b); }
bool isGreen() { float r,g,b; readColor(r,g,b); return (g > r && g > b); }
bool isBlue()  { float r,g,b; readColor(r,g,b); return (b > r && b > g); }

// End of file
