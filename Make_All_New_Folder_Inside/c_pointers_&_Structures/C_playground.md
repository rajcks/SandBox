---
marp: true
---

# C Playground
> Click **Run** above each snippet to compile and run it.

# C Programming Playground — From Basics to Pro in Embedded_C

This README contains runnable C examples from basic syntax to embedded programming concepts.  
Click **Run** above any code block (with the right VS Code extension + GCC) to execute.

---

## 📝 Topics Covered from basics to pro level:

1. **C Fundamentals** — variables, types, and functions  
2. **Pointers** — the single most important concept in C  
3. **Structures** — how they map to memory  
4. **Pointers + Structures** — using `->` and passing to functions  
5. **Memory layout, alignment, padding** — critical for embedded  
6. **Bitwise operations** — essential for hardware  
7. **Embedded I/O basics** — registers, peripherals, memory-mapped I/O  
8. **Data serialization** — packets, protocols  
9. **Best practices** — safety and portability

---

## 1️⃣ C Fundamentals – The Foundation
### Variables and types
Before embedded C, you must know regular C well.
```c
#include <stdio.h>
#include <stdint.h>  // fixed-width types

int main(void) {
    int a = 10;            // normal int (size depends on platform)
    uint8_t b = 255;       // unsigned 8-bit (portable)
    uint16_t c = 65535;    // unsigned 16-bit
    float d = 3.14;       // floating point

    printf("a=%d, b=%u, c=%u, d=%.2f\n", a, b, c, d);
    return 0;
}
```
> 💡 Embedded tip: Always prefer uint8_t, uint16_t, uint32_t over unsigned int so you know exactly how many bits you’re dealing with.

### Functions
```c
#include <stdio.h>

int add(int x, int y) {
    return x + y;
}

int main(void) {
    printf("%d\n", add(5, 3)); // 8
}
```
> 💡 Embedded tip: Keep functions small and focused. Avoid deep recursion to save stack space.

## 2️⃣ Pointers – Your Best Friend in C
A pointer is just an address of a variable in memory.
```c
#include <stdio.h>
#include <stdint.h>

int main(void) {
    int x = 42;
    int *p = &x;
    printf("x=%d\n", x);            // 42
    printf("&x=%d\n", (void*)&x);   // address of x
    printf("p=%d\n", (void*)p);     // value stored in p (address of x)
    printf("*p=%d\n", *p);          // value pointed to by p [42 (dereference pointer)]
    *p = 100;                       // change value through pointer
    printf("x=%d\n", x);            // 100

}
```
> 💡 Embedded tip: Pointers let you work directly with hardware registers and buffers.

### Pointers to arrays
```c
#include <stdio.h>
#include <stdint.h>

int main(void) {
    uint8_t  arr[4] = {10, 20, 30, 40};
    uint8_t  *p = arr;                 // points to first element
    printf("arr[0]=%d", arr[0]);
    printf("%u\n", *p);       // 1
    printf("%u\n", *(p + 1)); // 2
    printf("%u\n", *(p + 2)); // 3
    printf("%u\n", *(p + 3)); // 4
    *(p+1) = 99;                  // write via pointer
    printf("arr[1]=%d\n", arr[1]);
    return 0;
}
```
Pointer arithmetic moves by the size of the pointed-to type. 

## 3️⃣ Structures – Grouping Data
A struct groups multiple variables together.
```c
#include <stdio.h>
#include <stdint.h>

struct Point {
    int16_t x;
    int16_t y;
};

int main(void) {
    struct Point p1 = { 10, 20 };
    printf("x=%d, y=%d\n", p1.x, p1.y);
}
```
### e.g. Structures & Memory (layout/offsets)
```c
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>   // offsetof

struct Packet {
    uint8_t  header;   // 1 byte
    uint16_t length;   // 2 bytes (likely aligned → padding after header)
    uint8_t  flags;    // 1 byte
};

int main(void) {
    struct Packet pkt = {0xAA, 0x1234, 0x05};
    printf("sizeof(Packet)=%zu\n", sizeof(struct Packet));
    printf("offsets: header=%zu length=%zu flags=%zu\n",
           offsetof(struct Packet, header),
           offsetof(struct Packet, length),
           offsetof(struct Packet, flags));
    printf("values: header=0x%02X length=0x%04X flags=0x%02X\n",
           pkt.header, pkt.length, pkt.flags);
    return 0;
}
```

## 4️⃣ Pointers to structs ( -> & pass-by-pointer )
```c
#include <stdio.h>
#include <stdint.h>

struct Point {
    int16_t x;
    int16_t y;
};

int main(void) {
    struct Point p1 = { 10, 20 };
    struct Point *ptr = &p1;

    printf("x=%d\n", ptr->x); // same as (*ptr).x
}
```
> 💡 Embedded tip: Structures are great for representing hardware registers.

### e.g. Pointers to structs
```c
#include <stdio.h>
#include <stdint.h>

typedef struct { int16_t x, y; } Point;

void move(Point *p, int dx, int dy) {
    p->x += dx;    // same as (*p).x
    p->y += dy;
}

int main(void) {
    Point p = {10, -5};
    printf("Before: (%d,%d)\n", p.x, p.y);
    move(&p, 3, 7);
    printf("After:  (%d,%d)\n", p.x, p.y);
    return 0;
}
```

## 5️⃣ Structures in Memory
Structs are stored in contiguous memory.
```c
#include <stdio.h>
#include <stdint.h>

struct Data {
    uint8_t a;
    uint16_t b;
    uint8_t c;
};

int main(void) {
    struct Data d = { 1, 0x1234, 2 };
    printf("Size: %zu\n", sizeof(d));
}
```
Depending on alignment rules, there may be padding bytes — this matters in embedded systems because it changes how data is read/written to hardware or sent over communication protocols.

### e.g. Memory layout, alignment & padding
```c
#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

struct A { uint8_t a; uint32_t b; uint8_t c; }; // likely padding after a and at end
struct B { uint32_t b; uint8_t a; uint8_t c; }; // better packing by ordering

int main(void) {
    printf("alignof(uint32_t)=%zu\n", alignof(uint32_t));
    printf("sizeof(struct A)=%zu\n", sizeof(struct A));
    printf("sizeof(struct B)=%zu\n", sizeof(struct B));
    return 0;
}
```

## 6️⃣ Bitwise Operations – Core for Embedded
These let you work with individual bits in registers.
```c
#include <stdio.h>
#include <stdint.h>

int main(void) {
    uint8_t reg = 0b00001111;
    reg |= (1 << 7); // set bit 7
    reg &= ~(1 << 0); // clear bit 0
    printf("reg: 0x%02X\n", reg);
}
```
📌 Essential for:
- Turning LEDs on/off
- Reading sensor status flags
- Configuring hardware registers

## 7️⃣ Embedded I/O – Memory-Mapped Registers
In embedded, hardware is accessed via special addresses.
- Example (pseudo-code for STM32 GPIO):
```c
#define GPIOA_ODR (*(volatile uint32_t*)0x48000014)

int main(void) {
    GPIOA_ODR |= (1 << 5);  // set bit 5 (turn LED on)
}
```
- `volatile` tells the compiler “don’t optimize this away; it can change at any time” (important for hardware).

### Embedded I/O basics (simulated registers with volatile)
```c
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// Simulated memory-mapped registers (don't use raw addresses on PC)
volatile uint32_t GPIO_ODR; // output data register
volatile uint32_t GPIO_IDR; // input data register

#define LED_PIN   (1u << 5)

static inline void gpio_write(uint32_t mask, bool on) {
    if (on) GPIO_ODR |= mask;
    else    GPIO_ODR &= ~mask;
}

int main(void) {
    GPIO_IDR = 0; // pretend button not pressed
    gpio_write(LED_PIN, true);
    printf("GPIO_ODR=0x%08X\n", GPIO_ODR);
    gpio_write(LED_PIN, false);
    printf("GPIO_ODR=0x%08X\n", GPIO_ODR);

    if (GPIO_IDR & (1u << 0)) printf("Button pressed\n");
    else                      printf("Button not pressed\n");
    return 0;
}
```

## 8️⃣ Serialization – Packing Data into Buffers
This is exactly what your original code does.
```c
#include <stdio.h>
#include <stdint.h>

static void write_u16_le(uint8_t *buf, uint16_t v) {
    buf[0] = (uint8_t)(v & 0xFF);
    buf[1] = (uint8_t)(v >> 8);
}
static void write_u32_le(uint8_t *buf, uint32_t v) {
    buf[0]=(uint8_t)(v);
    buf[1]=(uint8_t)(v>>8);
    buf[2]=(uint8_t)(v>>16);
    buf[3]=(uint8_t)(v>>24);
}
static uint16_t read_u16_le(const uint8_t *buf) {
    return (uint16_t)buf[0] | ((uint16_t)buf[1] << 8);
}

int main(void) {
    uint8_t pkt[8] = {0};
    pkt[0] = 0xA5;                      // header
    write_u16_le(&pkt[1], 0x1234);      // length
    pkt[3] = 0x07;                      // flags
    write_u32_le(&pkt[4], 0xDEADBEEF);  // payload id

    printf("Packet:");
    for (size_t i = 0; i < sizeof pkt; ++i) printf(" %02X", pkt[i]);
    printf("\nlen(readback)=0x%04X\n", read_u16_le(&pkt[1]));
    return 0;
}

```

## 9️⃣ Combining Everything – Example Packet Builder
This is basically your original snippet, but fully explained:
```c
#include <stdint.h>

struct DST { uint8_t zone; uint8_t active; };
struct User { struct DST dst; int16_t utc_offset; };

uint32_t rtcGetTimeXT(void) { return 1735689600u; }

void Long_in_Feld(uint32_t v, uint8_t *dst) {
    dst[0] = (v >> 24) & 0xFF;
    dst[1] = (v >> 16) & 0xFF;
    dst[2] = (v >> 8)  & 0xFF;
    dst[3] = (v >> 0)  & 0xFF;
}

void Short_in_Feld(uint16_t v, uint8_t *dst) {
    dst[0] = (v >> 8) & 0xFF;
    dst[1] = (v >> 0) & 0xFF;
}

int main(void) {
    uint8_t out[16];
    size_t out_offset = 0;

    struct User u = { { 2, 1 }, 120 }; // zone=2, DST active, offset=+120 min

    Long_in_Feld(rtcGetTimeXT(), &out[out_offset]); out_offset += 4;
    out[out_offset++] = u.dst.zone;
    out[out_offset++] = u.dst.active;
    Short_in_Feld((uint16_t)u.utc_offset, &out[out_offset]); out_offset += 2;

    // Now `out` contains serialized data ready for transmission
}
```

### e.g. Best practices (fixed-width types, checks, _Static_assert)
```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>

enum Status { OK=0, ERR_RANGE=-1, ERR_NULL=-2 };

_Static_assert(sizeof(uint8_t) == 1, "uint8_t must be 1 byte");

int safe_sum_u8(const uint8_t *buf, size_t n, uint32_t *out) {
    if (!buf || !out) return ERR_NULL;
    uint32_t s = 0;
    for (size_t i = 0; i < n; ++i) s += buf[i];
    *out = s;
    return OK;
}

int bounded_copy(uint8_t *dst, size_t dst_cap, const uint8_t *src, size_t n) {
    if (!dst || !src) return ERR_NULL;
    if (n > dst_cap)  return ERR_RANGE;
    memcpy(dst, src, n);
    return OK;
}

int main(void) {
    uint8_t data[4] = {10, 20, 30, 40};
    uint32_t sum = 0;

    int rc = safe_sum_u8(data, 4, &sum);
    printf("sum rc=%d val=%u\n", rc, sum);

    uint8_t dst[3];
    rc = bounded_copy(dst, sizeof dst, data, 3);
    printf("copy rc=%d first=%u\n", rc, dst[0]);

    rc = bounded_copy(dst, sizeof dst, data, 4);
    printf("copy-too-long rc=%d\n", rc);
    return 0;
}
```