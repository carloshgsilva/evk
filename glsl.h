
struct vec2 {
    float x, y;
};

struct vec3 {
    float x, y, z;
};

struct vec4 {
    float x, y, z, w;
};

struct ivec2 {
    int x, y;
};

struct ivec3 {
    int x, y, z;
};

struct ivec4 {
    int x, y, z, w;
};

struct mat3 {
    float a[9];
};

struct mat4 {
    float a[16];

    vec3 right() {
        return vec3{a[0], a[1], a[2]};
    }
    vec3 up() {
        return vec3{a[4], a[5], a[6]};
    }
    vec3 forward() {
        return vec3{a[8], a[9], a[10]};
    }
};

struct quat : vec4 {
};

vec2 operator+(const vec2& a, const vec2& b) {
    return vec2(a.x + b.x, a.y + b.y);
}

vec3 operator+(const vec3& a, const vec3& b) {
    return vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}
vec3 operator-(const vec3& a, const vec3& b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}
vec3 operator*(const vec3& a, float b) {
    return vec3{a.x * b, a.y * b, a.z * b};
}

vec3 cross(const vec3& a, const vec3& b) {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
vec3 normalize(const vec3& a) {
    float l = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
    return vec3{a.x / l, a.y / l, a.z / l};
}

vec4 normalize(const vec4& a) {
    float l = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w);
    return quat{a.x / l, a.y / l, a.z / l, a.w / l};
}

float distance(const vec2& a, const vec2& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return sqrtf(dx*dx + dy*dy);
}
float distance(const vec3& a, const vec3& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

vec3 quat_rotate(const quat& q, const vec3& v) {
    vec3 xyz = vec3(q.x, q.y, q.z);
    vec3 t = cross(xyz, cross(xyz, v) + v * q.w);
    return v + t + t;
}

mat4 mat4_rotate(const vec3& a, float angle) {
    vec3 axis = normalize(a);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4{
        oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
        oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
        oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
        0.0,                                0.0,                                0.0,                                1.0
    };
}
void mat4_translate(mat4& m, const vec3& a) {
    m.a[12] += a.x;
    m.a[13] += a.y;
    m.a[14] += a.z;
}

uint32_t part1by2(uint32_t n) {
    n &= 0x000003ff;
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n << 8)) & 0x0300f00f;
    n = (n ^ (n << 4)) & 0x030c30c3;
    n = (n ^ (n << 2)) & 0x09249249;
    return n;
}
uint32_t unpart1by2(uint32_t n) {
    n &= 0x09249249;
    n = (n ^ (n >> 2)) & 0x030c30c3;
    n = (n ^ (n >> 4)) & 0x0300f00f;
    n = (n ^ (n >> 8)) & 0xff0000ff;
    n = (n ^ (n >> 16)) & 0x000003ff;
    return n;
}

uint32_t morton32_encode(uint32_t x, uint32_t y, uint32_t z){
    return part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2);
}

void morton32_decode(uint32_t n, uint32_t& x, uint32_t& y, uint32_t& z){
    x = unpart1by2(n);
    y = unpart1by2(n >> 1);
    z = unpart1by2(n >> 2);
}