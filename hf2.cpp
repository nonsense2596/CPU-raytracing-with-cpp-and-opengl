//=============================================================================================
// Mintaprogram: Computer Graphics Sample Program: Ray-tracing-let
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Akari Saito 
// Neptun : WHMSCH
// https://na.finalfantasyxiv.com/lodestone/character/6712899/
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const float epsilon = 0.001f; // default: 0.0001f

// a fordito kerte, hogy class legyen
enum class MaterialType { 
	ROUGH, 
	REFLECTIVE 
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

float rnd() { 
	return (float)rand() / RAND_MAX; 
}

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(MaterialType::ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(MaterialType::REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) /
			((n + one) * (n + one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Quadrics : public Intersectable {
	mat4 Q;
	float zmin, zmax;
	vec3 translation;

	Quadrics(mat4& _Q, float _zmin, float _zmax, vec3 _translation, Material* _material) {
		Q = _Q; zmin = _zmin; zmax = _zmax;
		translation = _translation;
		material = _material;
	}
	vec3 gradf(vec3 r) {
		vec4 g = vec4(r.x, r.y, r.z, 1) * Q * 2;
		return vec3(g.x, g.y, g.z);
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 start = ray.start - translation;
		vec4 S(start.x, start.y, start.z, 1), D(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		float a = dot(D * Q, D), b = dot(S * Q, D) * 2, c = dot(S * Q, S);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);

		float t1 = (-b + sqrt_discr) / 2.0f / a;
		vec3 p1 = ray.start + ray.dir * t1;
		if (p1.z < zmin || p1.z > zmax) t1 = -1;

		float t2 = (-b - sqrt_discr) / 2.0f / a;
		vec3 p2 = ray.start + ray.dir * t2;
		if (p2.z < zmin || p2.z > zmax) t2 = -1;

		if (t1 <= 0 && t2 <= 0) return hit;
		if (t1 <= 0) hit.t = t2;
		else if (t2 <= 0) hit.t = t1;
		else if (t2 < t1) hit.t = t2;
		else hit.t = t1;
		hit.position = start + ray.dir * hit.t;
		hit.normal = normalize(gradf(hit.position));
		hit.position = hit.position + translation;
		hit.material = material;
		return hit;
	}
};

struct Pentagon : Intersectable {
	vec3 p1, p2, p3, p4, p5;
	vec3 n;
	Pentagon(vec3 _p1, vec3 _p2, vec3 _p3, vec3 _p4, vec3 _p5, Material* _material, float _scale = 1.0f, vec3 _offset = vec3(0.0f,0.0f,0.0f)) 
	{
		material = _material;
		vec3 offset = vec3(_offset);
		p1 = _scale*(_p1+offset); 
		p2 = _scale*(_p2+offset); 
		p3 = _scale*(_p3+offset); 
		p4 = _scale*(_p4+offset); 
		p5 = _scale*(_p5+offset);

		n = normalize(cross(p2 - p1, p5 - p1));	
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		// sikmetszes t parametere
		float t = (dot(p1 - ray.start, n) / dot(ray.dir, n));
		if (t < 0.0f) 
		{
			return hit;
		}
		// sikmetszes pozicioja vec3
		vec3 hitpos = ray.start + ray.dir * t;
		// megvizsgaljuk hogy a pentagonon kivul esik, akkor nem metszi tehat kilepunk
		if (!isInside(hitpos)) 
		{
			return hit;
		}
		// ha belul esik akkor...
		hit.position = hitpos;
		hit.material = material;
		hit.normal = n;
		hit.t = t;
		return hit;
	}
	bool isInside(vec3 hitpos) {
		if (dot(cross(p2 - p1, hitpos - p1), n) < 0.0f)
		{
			return false;
		}
		if (dot(cross(p3 - p2, hitpos - p2), n) < 0.0f) 
		{
			return false;
		}
		if (dot(cross(p4 - p3, hitpos - p3), n) < 0.0f) 
		{
			return false;
		}
		if (dot(cross(p5 - p4, hitpos - p4), n) < 0.0f) 
		{
			return false;
		}
		if (dot(cross(p1 - p5, hitpos - p5), n) < 0.0f) 
		{
			return false;
		}
		return true;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float dt) 
	{
		vec3 d = eye - lookat;
		eye = vec3( d.x,
					d.y*cos(dt)+d.z*sin(dt),
					d.y*(-sin(dt))+d.z*cos(dt)
					) + lookat;
		float fov = 45 * M_PI / 180; 
		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 location;
	vec3 Le;
	Light(vec3 _location, vec3 _Le) {
		location = _location;
		Le = _Le;
	}
	float distanceOf(vec3 point) {
		return length(location - point);
	}
	vec3 directionOf(vec3 point) {
		return normalize(location - point);	// normalize?
	}
	vec3 radianceAt(vec3 point) {
		float distance2 = dot(location - point, location - point);
		if (distance2 < epsilon) distance2 = epsilon;
		return Le / distance2 / 4 / M_PI;
	}

};

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {

		vec3 eye = vec3(0, 0, 1.6f);
		vec3 vup = vec3(0, 1.2f, 0);
		vec3 lookat = vec3(0, 0, -0.125f);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);

		lights.push_back(
			new Light(
				vec3(2.0f, 2.0f, 3.0f), 
				vec3(1000.0f, 1000.0f, 1000.0f)
			)
		);

		lights.push_back(
			new Light(
				vec3(0.2f,  0.0f, 0.0f), 
				vec3(1.0f, 1.0f, 1.0f)
			)
		);


		vec3 kd_default(0.3f, 0.2f, 0.1f);
		vec3 ks_default(2, 2, 2);
		Material* roughmaterial = new RoughMaterial(
			kd_default, 
			ks_default, 
			30
		);

		vec3 n_default(1, 1, 1);
		vec3 kappa_default(5, 4, 3);
		Material* reflectivematerial = new ReflectiveMaterial(
			n_default, 
			kappa_default
		);

		
		vec3 kd_pink(0.7f, 0.2f, 0.5f);
		vec3 ks_pink(2.0f, 2.0f, 2.0f);
		Material* pink = new RoughMaterial(
			kd_pink,
			ks_pink, 
			50
		);


		vec3 n_gold(0.17f, 0.35f, 1.5f);
		vec3 kappa_gold(3.1f, 2.7f, 1.9f);
		Material* gold = new ReflectiveMaterial(
			n_gold, 
			kappa_gold
		);


		mat4 paraboloidd1 = mat4(1.0f, 0, 0, 0,
								 0, 1.0f, 0, 0,
								 0, 0, 0, 0.5f,
								 0, 0, 0.5f, -0.125);
		objects.push_back(
			new Quadrics(
				paraboloidd1, 
				0, 
				0.122f, 
				vec3(0,0,0), 
				gold
			)
		);

		mat4 paraboloidd2 = mat4(-1.0f, 0, 0, 0,
								 0, -1.0f, 0, 0,
								 0, 0, 0, 0.5f,
								 0, 0, 0.5f, 0.125);
		objects.push_back(
			new Quadrics(
				paraboloidd2, 
				-0.125f, 
				0, 
				vec3(0, 0, 0), 
				gold
			)
		);

		// belso
																																																		 // scale, offset
		objects.push_back(new Pentagon(vec3(0.0f, 0.618f, 1.618f),	vec3(0.0f, -0.618f, 1.618f),	vec3(1.0f, -1.0f, 1.0f),		vec3(1.618f, 0.0f, 0.618f),		vec3(1.0f, 1.0f, 1.0f),			pink,	0.01f, vec3(0.0f,0.0f,-10.9f)));
		objects.push_back(new Pentagon(vec3(0.0f, 0.618f, 1.618f),	vec3(1.0f, 1.0f, 1.0f),			vec3(0.618f, 1.618f, 0.0f),		vec3(-0.618f, 1.618f, 0.0f),	vec3(-1.0f, 1.0f, 1.0f),		pink,	0.01f, vec3(0.0f,0.0f,-10.9f)));
		objects.push_back(new Pentagon(vec3(0.0f, 0.618f, 1.618f),	vec3(-1.0f, 1.0f, 1.0f),		vec3(-1.618f, 0.0f, 0.618f),	vec3(-1.0f, -1.0f, 1.0f),		vec3(0.0f, -0.618f, 1.618f),	pink,	0.01f, vec3(0.0f,0.0f,-10.9f)));
		objects.push_back(new Pentagon(vec3(0.0f, -0.618f, 1.618f), vec3(-1.0f, -1.0f, 1.0f),		vec3(-0.618f, -1.618f, 0.0f),	vec3(0.618f, -1.618f, 0.0f),	vec3(1.0f, -1.0f, 1.0f),		pink,	0.01f, vec3(0.0f,0.0f,-10.9f)));
		objects.push_back(new Pentagon(vec3(0.0f, -0.618f, -1.618f),vec3(0.0f, 0.618f, -1.618f),	vec3(1.0f, 1.0f, -1.0f),		vec3(1.618f, 0.0f, -0.618f),	vec3(1.0f, -1.0f, -1.0f),		pink,	0.01f, vec3(0.0f,0.0f,-10.9f)));
		objects.push_back(new Pentagon(vec3(0.0f, -0.618f, -1.618f),vec3(1.0f, -1.0f, -1.0f),		vec3(0.618f, -1.618f, 0.0f),	vec3(-0.618f, -1.618f, 0.0f),	vec3(-1.0f, -1.0f, -1.0f),		pink,	0.01f, vec3(0.0f,0.0f,-10.9f)));
		objects.push_back(new Pentagon(vec3(0.0f, -0.618f, -1.618f),vec3(-1.0f, -1.0f, -1.0f),		vec3(-1.618f, 0.0f, -0.618f),	vec3(-1.0f, 1.0f, -1.0f),		vec3(0.0f, 0.618f, -1.618f),	pink,	0.01f, vec3(0.0f,0.0f,-10.9f)));
		objects.push_back(new Pentagon(vec3(-1.0f, 1.0f, -1.0f),	vec3(-0.618f, 1.618f, 0.0f),	vec3(0.618f, 1.618f, 0.0f),		vec3(1.0f, 1.0f, -1.0f),		vec3(0.0f, 0.618f, -1.618f),	pink,	0.01f, vec3(0.0f,0.0f,-10.9f)));
		objects.push_back(new Pentagon(vec3(1.0f, -1.0f, 1.0f),		vec3(0.618f, -1.618f, 0.0f),	vec3(1.0f, -1.0f, -1.0f),		vec3(1.618f, 0.0f, -0.618f),	vec3(1.618f, 0.0f, 0.618f),		pink,	0.01f, vec3(0.0f,0.0f,-10.9f)));
		objects.push_back(new Pentagon(vec3(1.618f, 0.0f, 0.618f),	vec3(1.618f, 0.0f, -0.618f),	vec3(1.0f, 1.0f, -1.0f),		vec3(0.618f, 1.618f, 0.0f),		vec3(1.0f, 1.0f, 1.0f),			pink,	0.01f, vec3(0.0f,0.0f,-10.9f)));
		objects.push_back(new Pentagon(vec3(-1.0f, 1.0f, 1.0f),		vec3(-0.618f, 1.618f, 0.0f),	vec3(-1.0f, 1.0f, -1.0f),		vec3(-1.618f, 0.0f, -0.618f),	vec3(-1.618f, 0.0f, 0.618f),	pink,	0.01f, vec3(0.0f,0.0f,-10.9f)));
		objects.push_back(new Pentagon(vec3(-1.618f, 0.0f, 0.618f), vec3(-1.618f, 0.0f, -0.618f),	vec3(-1.0f, -1.0f, -1.0f),		vec3(-0.618f, -1.618f, 0.0f),	vec3(-1.0f, -1.0f, 1.0f),		pink,	0.01f, vec3(0.0f,0.0f,-10.9f)));

		// kulso
		
		objects.push_back(new Pentagon(vec3(0.0f, 6.18f, 16.18f),	vec3(0.0f, -6.18f, 16.18f),		vec3(10.0f, -10.0f, 10.0f),		vec3(16.18f, 0.0f, 6.18f),		vec3(10.0f, 10.0f, 10.0f),		roughmaterial, 1.0f));
		objects.push_back(new Pentagon(vec3(0.0f, 6.18f, 16.18f),	vec3(10.0f, 10.0f, 10.0f),		vec3(6.18f, 16.18f, 0.0f),		vec3(-6.18f, 16.18f, 0.0f),		vec3(-10.0f, 10.0f, 10.0f),		roughmaterial, 1.0f));
		objects.push_back(new Pentagon(vec3(0.0f, 6.18f, 16.18f),	vec3(-10.0f, 10.0f, 10.0f),		vec3(-16.18f, 0.0f, 6.18f),		vec3(-10.0f, -10.0f, 10.0f),	vec3(0.0f, -6.18f, 16.18f),		roughmaterial, 1.0f));
		objects.push_back(new Pentagon(vec3(0.0f, -6.18f, 16.18f),	vec3(-10.0f, -10.0f, 10.0f),	vec3(-6.18f, -16.18f, 0.0f),	vec3(6.18f, -16.18f, 0.0f),		vec3(10.0f, -10.0f, 10.0f),		roughmaterial, 1.0f));
		objects.push_back(new Pentagon(vec3(0.0f, -6.18f, -16.18f), vec3(0.0f, 6.18f, -16.18f),		vec3(10.0f, 10.0f, -10.0f),		vec3(16.18f, 0.0f, -6.18f),		vec3(10.0f, -10.0f, -10.0f),	roughmaterial, 1.0f));
		objects.push_back(new Pentagon(vec3(0.0f, -6.18f, -16.18f), vec3(10.0f, -10.0f, -10.0f),	vec3(6.18f, -16.18f, 0.0f),		vec3(-6.18f, -16.18f, 0.0f),	vec3(-10.0f, -10.0f, -10.0f),	roughmaterial, 1.0f));
		objects.push_back(new Pentagon(vec3(0.0f, -6.18f, -16.18f), vec3(-10.0f, -10.0f, -10.0f),	vec3(-16.18f, 0.0f, -6.18f),	vec3(-10.0f, 10.0f, -10.0f),	vec3(0.0f, 6.18f, -16.18f),		roughmaterial, 1.0f));
		objects.push_back(new Pentagon(vec3(-10.0f, 10.0f, -10.0f), vec3(-6.18f, 16.18f, 0.0f),		vec3(6.18f, 16.18f, 0.0f),		vec3(10.0f, 10.0f, -10.0f),		vec3(0.0f, 6.18f, -16.18f),		roughmaterial, 1.0f));
		objects.push_back(new Pentagon(vec3(10.0f, -10.0f, 10.0f),	vec3(6.18f, -16.18f, 0.0f),		vec3(10.0f, -10.0f, -10.0f),	vec3(16.18f, 0.0f, -6.18f),		vec3(16.18f, 0.0f, 6.18f),		roughmaterial, 1.0f));
		objects.push_back(new Pentagon(vec3(16.18f, 0.0f, 6.18f),	vec3(16.18f, 0.0f, -6.18f),		vec3(10.0f, 10.0f, -10.0f),		vec3(6.18f, 16.18f, 0.0f),		vec3(10.0f, 10.0f, 10.0f),		roughmaterial, 1.0f));
		objects.push_back(new Pentagon(vec3(-10.0f, 10.0f, 10.0f),	vec3(-6.18f, 16.18f, 0.0f),		vec3(-10.0f, 10.0f, -10.0f),	vec3(-16.18f, 0.0f, -6.18f),	vec3(-16.18f, 0.0f, 6.18f),		roughmaterial, 1.0f));
		objects.push_back(new Pentagon(vec3(-16.18f, 0.0f, 6.18f),	vec3(-16.18f, 0.0f, -6.18f),	vec3(-10.0f, -10.0f, -10.0f),	vec3(-6.18f, -16.18f, 0.0f),	vec3(-10.0f, -10.0f, 10.0f),	roughmaterial, 1.0f));


	}


	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); 
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}


	vec3 trace(Ray ray, int depth = 0) {

		if (depth > 5) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance(0, 0, 0);

		if (hit.material->type == MaterialType::ROUGH) 
		{
			outRadiance = hit.material->ka * La;
			for (Light* light : lights) 
			{
				// hogy ne arnyekolja le egy fenynel messzebb levo test a fenyt, itt is firstIntersect-et hasznalunk
				vec3 outDirection = light->directionOf(hit.position);
				Ray shadowRay(hit.position + hit.normal * epsilon, outDirection);
				if ( (firstIntersect(shadowRay).t > light->distanceOf(hit.position)) || (firstIntersect(shadowRay).t < epsilon) ) 
				{
					float cosTheta = dot(hit.normal, outDirection);
					if (cosTheta > epsilon) 
					{
						outRadiance = outRadiance + light->radianceAt(hit.position) * hit.material->kd * cosTheta;

						vec3 halfway = normalize(-ray.dir + light->directionOf(hit.position));
						float cosDelta = dot(hit.normal, halfway);
						if (cosDelta > 0) 
						{
							outRadiance = outRadiance + light->radianceAt(hit.position) * hit.material->ks * powf(cosDelta, hit.material->shininess);
						}
					}
				}
			}
		}
		if (hit.material->type == MaterialType::REFLECTIVE) 
		{
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1.0f, 1.0f, 1.0f);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1.0f - cosa, 5.0f);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}

		return outRadiance;
	}
	void Animate(float dt) 
	{
		camera.Animate(dt);
	}
};




GPUProgram gpuProgram; 
Scene scene;

const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// [-1,1] to [0,1]
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)"; 

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	
		glBindVertexArray(vao);		
		unsigned int vbo;		
		glGenBuffers(1, &vbo);	
		glBindBuffer(GL_ARRAY_BUFFER, vbo); 
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_DYNAMIC_DRAW);	   
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     
	}
	void Draw() {
		glBindVertexArray(vao);	
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	
	}
};
FullScreenTexturedQuad* fullScreenTexturedQuad;
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'f') {
		scene.Animate(0.3f);
	}
	if (key == 'F') {
		scene.Animate(-0.3f);
	}
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
	
}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	//scene.Animate(0.1f);
	glutPostRedisplay();
}


/* NON-QUADRATIC FORM OF PARABOLOIDS
struct Paraboloid : Intersectable {

	Paraboloid(Material* _material) {
		material = _material;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		float u1 = ray.dir.x;
		float u2 = ray.dir.z;
		float u3 = ray.dir.y;
		float x0 = ray.start.x;
		float y0 = ray.start.z;
		float z0 = ray.start.y;
		// ket lambda lesz
		float discr = (2.0f * (u1 * x0 + u2 * y0) - u3) * (2.0f * (u1 * x0 + u2 * y0) - u3) - (4.0f * (u1 * u1 + u2 * u2) * (x0 * x0 + y0 * y0 - z0 - (1.0f / 8.0f)));
		// ha gyok alatt negativ, nincs eredmeny es return hit;
		if (discr < 0.0f) return hit;
		float lambda1 = (-(2.0f * (u1 * x0 + u2 * y0) - u3) + sqrtf(discr)) / (2.0f * (u1 * u1 + u2 * u2));
		float lambda2 = (-(2.0f * (u1 * x0 + u2 * y0) - u3) - sqrtf(discr)) / (2.0f * (u1 * u1 + u2 * u2));
		// kisebb kell
		float lambda;
		if (lambda2 < lambda1) lambda = lambda2;
		else lambda = lambda1;
		// ezek a metszespont koordinatai
		float x = u1 * lambda + x0;
		float y = u2 * lambda + y0;
		float z = u3 * lambda + z0;
		// z feltetelvizsgalat, -1/8 es 0 kozott
		if (z > 0.0f || z < -1.8f) return hit;
		// normalvektor a gradiense az implicit alaknak, x,y,z ertekekkel behelyettesitve es normalizalva
		// a = b = 1, [2x/aa, 2y/bb, -1]
		vec3 n = normalize(vec3(2.0f * x,
			2.0f * y,
			-1.0f));
		// t ????
		// output
		hit.position = vec3(x, y, z);

		hit.material = material;
		hit.normal = n;
		hit.t = lambda;
		return hit;
	}
};*/


