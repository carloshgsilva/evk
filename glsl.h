
using uint = unsigned int;

struct vec2 {
    float x, y;

    vec2() = default;
    vec2(float v) : x(v), y(v) {}
    vec2(float x, float y) : x(x), y(y) {};
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

struct vec3 {
    float x, y, z;
    
    vec3() = default;
    vec3(float v) : x(v), y(v), z(v) {}
    vec3(float x, float y, float z) : x(x), y(y), z(z) {};
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

struct vec4 {
    float x, y, z, w;
    
    vec4() = default;
    vec4(float v) : x(v), y(v), z(v), w(w) {}
    vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {};
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
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
    union {
        float a[16];
        float m[4][4];
        vec4 v[4];
    };

    vec3 right() {
        return vec3{a[0], a[1], a[2]};
    }
    vec3 up() {
        return vec3{a[4], a[5], a[6]};
    }
    vec3 forward() {
        return vec3{a[8], a[9], a[10]};
    }

    vec4& operator[](int i) { return v[i]; }
    const vec4& operator[](int i) const { return v[i]; }
};

struct quat : public vec4 {
    quat() = default;
    quat(float v) : vec4(v) {}
    quat(float x, float y, float z, float w) : vec4(x, y, z, w) {};
};

#define GLSL_OP_VEC2(_type, _op) \
_type operator##_op(const _type& a, const _type& b) { return _type(a.x _op b.x, a.y _op b.y); }
#define GLSL_OP_VEC3(_type, _op) \
_type operator##_op(const _type& a, const _type& b) { return _type(a.x _op b.x, a.y _op b.y, a.z _op b.z); }
#define GLSL_OP_VEC4(_type, _op) \
_type operator##_op(const _type& a, const _type& b) { return _type(a.x _op b.x, a.y _op b.y, a.z _op b.z, a.w _op b.w); }

#define GLSL_OP(_op)    \
GLSL_OP_VEC2(vec2, _op) \
GLSL_OP_VEC3(vec3, _op) \
GLSL_OP_VEC4(vec4, _op)

GLSL_OP(*);
GLSL_OP(+);
GLSL_OP(-);
GLSL_OP(/);

vec3 cross(const vec3& a, const vec3& b) {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

vec2 normalize(const vec2& a) {
    float l = sqrtf(a.x*a.x + a.y*a.y);
    return vec2(a.x / l, a.y / l);
}
vec3 normalize(const vec3& a) {
    float l = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
    return vec3(a.x / l, a.y / l, a.z / l);
}
vec4 normalize(const vec4& a) {
    float l = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w);
    return quat(a.x / l, a.y / l, a.z / l, a.w / l);
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
float distance(const vec4& a, const vec4& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    float dw = a.w - b.w;
    return sqrtf(dx*dx + dy*dy + dz*dz + dw*dw);
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

// http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
quat mat4_to_quat(const mat4& m) {
    float tr = m[0][0] + m[1][1] + m[2][2];
    quat q = quat(0.0f, 0.0f, 0.0f, 0.0f);

    if (tr > 0)
    {
        float s = sqrt(tr + 1.0) * 2; // S=4*qw 
        q.w = 0.25 * s;
        q.x = (m[2][1] - m[1][2]) / s;
        q.y = (m[0][2] - m[2][0]) / s;
        q.z = (m[1][0] - m[0][1]) / s;
    }
    else if ((m[0][0] > m[1][1]) && (m[0][0] > m[2][2]))
    {
        float s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2; // S=4*qx 
        q.w = (m[2][1] - m[1][2]) / s;
        q.x = 0.25 * s;
        q.y = (m[0][1] + m[1][0]) / s;
        q.z = (m[0][2] + m[2][0]) / s;
    }
    else if (m[1][1] > m[2][2])
    {
        float s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2; // S=4*qy
        q.w = (m[0][2] - m[2][0]) / s;
        q.x = (m[0][1] + m[1][0]) / s;
        q.y = 0.25 * s;
        q.z = (m[1][2] + m[2][1]) / s;
    }
    else
    {
        float s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2; // S=4*qz
        q.w = (m[1][0] - m[0][1]) / s;
        q.x = (m[0][2] + m[2][0]) / s;
        q.y = (m[1][2] + m[2][1]) / s;
        q.z = 0.25 * s;
    }

    return quat(q);
}
mat4 quat_to_mat4(const quat& q) {
    mat4 m = mat4();

    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    m[0][0] = 1.0 - (yy + zz);
    m[0][1] = xy - wz;
    m[0][2] = xz + wy;

    m[1][0] = xy + wz;
    m[1][1] = 1.0 - (xx + zz);
    m[1][2] = yz - wx;

    m[2][0] = xz - wy;
    m[2][1] = yz + wx;
    m[2][2] = 1.0 - (xx + yy);

    m[3][3] = 1.0;

    return m;
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