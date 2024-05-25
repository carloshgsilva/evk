
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
        vec4 v[4];
    };

    mat4() {
        for(int i = 0; i < 16; i++) {
            a[i] = 0.0f;
        }
    }
    mat4(float d) : mat4() {
        for(int i = 0; i < 4; i++) {
            a[i*5] = 1.0f;
        }
    }
    mat4(const vec4& v0, const vec4& v1, const vec4& v2, const vec4& v3) {
        v[0] = v0;
        v[1] = v1;
        v[2] = v2;
        v[3] = v3;
    }
    mat4(float e0, float e1, float e2, float e3, float e4, float e5, float e6, float e7, float e8, float e9, float e10, float e11, float e12, float e13, float e14, float e15) {
        a[0] = e0;
        a[1] = e1;
        a[2] = e2;
        a[3] = e3;
        a[4] = e4;
        a[5] = e5;
        a[6] = e6;
        a[7] = e7;
        a[8] = e8;
        a[9] = e9;
        a[10] = e10;
        a[11] = e11;
        a[12] = e12;
        a[13] = e13;
        a[14] = e14;
        a[15] = e15;
    }

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


float dot(const vec2& a, const vec2& b) {
    return a.x*b.x + a.y*b.y;
}
float dot(const vec3& a, const vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
float dot(const vec4& a, const vec4& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

vec3 cross(const vec3& a, const vec3& b) {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float length(const vec2& a) {
    float dx = a.x;
    float dy = a.y;
    return sqrtf(dx*dx + dy*dy);
}
float length(const vec3& a) {
    float dx = a.x;
    float dy = a.y;
    float dz = a.z;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}
float length(const vec4& a) {
    float dx = a.x;
    float dy = a.y;
    float dz = a.z;
    float dw = a.w;
    return sqrtf(dx*dx + dy*dy + dz*dz + dw*dw);
}

vec2 normalize(const vec2& a) {
    return a / length(a);
}
vec3 normalize(const vec3& a) {
    return a / length(a);
}
vec4 normalize(const vec4& a) {
    return a / length(a);
}
quat normalize(const quat& a) {
    float l = length(a);
    return quat(a.x / l, a.y / l, a.z / l, a.w / l);
}

float distance(const vec2& a, const vec2& b) {
    return length(a - b);
}
float distance(const vec3& a, const vec3& b) {
    return length(a - b);
}
float distance(const vec4& a, const vec4& b) {
    return length(a - b);
}

float determinant(const mat4& m) {
    float n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
    float n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
    float n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
    float n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

    float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    return n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
}
mat4 inverse(const mat4& m) {
    float n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
    float n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
    float n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
    float n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

    float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    float idet = 1.0f / det;

    mat4 ret;

    ret[0][0] = t11 * idet;
    ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
    ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
    ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

    ret[1][0] = t12 * idet;
    ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
    ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
    ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

    ret[2][0] = t13 * idet;
    ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
    ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
    ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

    ret[3][0] = t14 * idet;
    ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
    ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
    ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

    return ret;
}

mat4 operator*(const mat4& a, const mat4& b) {
    mat4 t = {};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float v = 0.0f;
            for (int k = 0; k < 4; k++) {
                v += a.a[k * 4 + j] * b.a[i * 4 + k];
            }
            t.a[i * 4 + j] = v;
        }
    }
    return t;
}

quat operator*(const quat& a, const quat& b) {
    return quat{
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w 
    };
}
vec3 operator*(const quat& q, const vec3& v) {
    vec3 xyz = vec3(q.x, q.y, q.z);
    vec3 t = cross(xyz, cross(xyz, v) + v * q.w);
    return v + t + t;
}
quat quat_from_axis_angle(const vec3& axis, float angle) {
   float factor = sinf( angle / 2.0f );

   quat q;
   q.x = axis.x * factor;
   q.y = axis.y * factor;
   q.z = axis.z * factor;

   q.w = cosf(angle/2.0f);
   return normalize(q);
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
void mat4_scale(mat4& m, const vec3& a) {
    m.a[0] *= a.x;
    m.a[1] *= a.y;
    m.a[2] *= a.z;
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


mat4 mat4_compose(const vec3& position, const quat& rotation, const vec3& scale) {
    mat4 m = quat_to_mat4(rotation);
    mat4_scale(m, scale);
    mat4_translate(m, position);
    return m;
}
void mat4_decompose(mat4 m, vec3& position, quat& rotation, vec3& scale) {
    float sx = length(vec3(m[0][0], m[0][1], m[0][2]));
    float sy = length(vec3(m[1][0], m[1][1], m[1][2]));
    float sz = length(vec3(m[2][0], m[2][1], m[2][2]));

    // if determine is negative, we need to invert one scale
    float det = determinant(m);
    if (det < 0) {
        sx = -sx;
    }

    position.x = m[3][0];
    position.y = m[3][1];
    position.z = m[3][2];

    // scale the rotation part
    float invSX = 1.0 / sx;
    float invSY = 1.0 / sy;
    float invSZ = 1.0 / sz;

    m[0][0] *= invSX;
    m[0][1] *= invSX;
    m[0][2] *= invSX;

    m[1][0] *= invSY;
    m[1][1] *= invSY;
    m[1][2] *= invSY;

    m[2][0] *= invSZ;
    m[2][1] *= invSZ;
    m[2][2] *= invSZ;

    rotation = mat4_to_quat(m);

    scale.x = sx;
    scale.y = sy;
    scale.z = sz;
}

mat4 mat4_look_at(const vec3& at, const vec3& eye, const vec3& up) {
    const vec3 f = normalize(at - eye);
    const vec3 s = normalize(cross(f, up));
    const vec3 u = cross(s, f);

    mat4 m = mat4(1.0f);
    m[0][0] = s.x;
    m[1][0] = s.y;
    m[2][0] = s.z;
    m[0][1] = u.x;
    m[1][1] = u.y;
    m[2][1] = u.z;
    m[0][2] =-f.x;
    m[1][2] =-f.y;
    m[2][2] =-f.z;
    m[3][0] =-dot(s, eye);
    m[3][1] =-dot(u, eye);
    m[3][2] = dot(f, eye);

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