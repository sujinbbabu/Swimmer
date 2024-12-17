#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using Vector = Eigen::Vector2d;
using Vector3 = Eigen::Vector3d;
using Tensor = Eigen::Matrix3d;

// Particle class
constexpr long double Pi = (3.1415926535897931L);

class Particle
{
private:
public:
  Particle();
  Particle(Vector pos, Vector vel);
  ~Particle() = default;
  Vector pos;
  Vector vel;
  Vector rvel = {0.0, 0.0};
  Vector rpos = {0.0, 0.0};
  std::array<int, 2> cell;
  friend std::ostream &operator<<(std::ostream &os, const Particle &p);
};

std::ostream &operator<<(std::ostream &os, const Particle &p)
{
  os << "Position " << p.pos[0] << " " << p.pos[1] << "\nVelocity " << p.vel[0]
     << " " << p.vel[1];
  return os;
}
Particle::Particle() : pos({0.0, 0.0}), vel({0.0, 0.0}) {}
Particle::Particle(Vector pos, Vector vel) : pos(pos), vel(vel) {}

// Utils
double sq(double a) { return a * a; }

bool overlap(int i, int j, double a, Vector spos, double radius)
{
  double d1 = sq(i * a - spos[0]) + sq(j * a - spos[1]);
  double d2 = sq(i * a + a - spos[0]) + sq(j * a - spos[1]);
  double d3 = sq(i * a - spos[0]) + sq(j * a + a - spos[1]);
  double d4 = sq(i * a + a - spos[0]) + sq(j * a + a - spos[1]);

  double dx = std::sqrt(std::max({d1, d2, d3, d4}));
  double dn = std::sqrt(std::min({d1, d2, d3, d4}));

  return (dx >= radius) && (dn <= radius);
}

bool insideCircle(const Vector &pos, const Vector &spos, const double radius)
{
  double d = (pos - spos).norm();
  return (std::sqrt(d) <= radius);
}

// -------------------//
//                    //
//  Geometry Class    //
//                    //
//--------------------//

double minDistLinePoint(Vector v, Vector w, Vector p)
{
  // Return minimum distance between line segment vw and point p
  const double l2 = (v - w).squaredNorm(); // i.e. |w-v|^2 -  avoid a sqrt
  if (l2 == 0.0)
    return (p - v).norm(); // v == w case
  // Consider the line extending the segment, parameterized as v + t (w - v).
  // We find projection of point p onto the line.
  // It falls where t = [(p-v) . (w-v)] / |w-v|^2
  // We clamp t from [0,1] to handle points outside the segment vw.
  const double pl2 = (p - v).dot(w - v) / l2;
  const float t = std::max(0.0, std::min(1.0, pl2));
  const Vector projection = v + t * (w - v); // Projection falls on the segment
  return (p - projection).norm();
}

bool ccw(Vector a, Vector b, Vector c)
{
  return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]);
}

bool lineSegmentsIntersect(Vector a, Vector b, Vector c, Vector d)
{
  return (ccw(a, c, d) != ccw(b, c, d)) && (ccw(a, b, c) != ccw(a, b, d));
}

bool linesIntersect(Vector p1, Vector p2, Vector q1, Vector q2, Vector &point)
{
  Vector s = p1 - p2;
  Vector r = q1 - q2;

  double denominator = s[0] * r[1] - s[1] * r[0];
  if (std::abs(denominator) < 1e-8)
  {
    return false;
  }

  double t = (r[1] * (p1[0] - q1[0]) - r[0] * (p1[1] - q1[1])) / denominator;
  if (t >= 0 && t <= 1)
  {
    point = p1 - t * s;
    return true;
  }
  return false;
}

class LineGeomtry
{
private:
public:
  Vector a;
  Vector b;

  LineGeomtry(Vector a, Vector b) : a(a), b(b) {}
  void fill(std::vector<Particle> &vparticles) {}
  bool collisionDetection(Vector ppos)
  {
    double d = minDistLinePoint(a, b, ppos);
    return d == 0;
  }
  void collide(Particle &particle, double dt)
  {
    Vector ppos = particle.pos;
    Vector oldpos = ppos - dt * particle.vel;
    if (lineSegmentsIntersect(a, b, oldpos, ppos))
    {
      if (linesIntersect(a, b, oldpos, ppos, ppos))
      {
        double tc = (ppos - oldpos).norm() / particle.vel.norm();
        particle.vel = -particle.vel;
        particle.pos = ppos + (dt - tc) * particle.vel;
      }
    }
  }
};

class CircleGeometry
{
private:
  std::uniform_real_distribution<double> radialDist;
  std::uniform_real_distribution<double> thetaDist;
  std::mt19937_64 randEng;
  int offset;
  int N = 0;

public:
  Vector pos;
  double radius;

  CircleGeometry(Vector pos, double r, std::vector<Particle> &vparticles,
                 int rho, double a, std::mt19937_64 gen)
      : randEng(gen), pos(pos), radius(r)
  {
    double ra = radius - std::sqrt(2) * a;
    radialDist = std::uniform_real_distribution<double>(ra, radius);
    thetaDist = std::uniform_real_distribution<double>(0, 2 * Pi);
    offset = vparticles.size();
    N = static_cast<int>(std::round(rho * Pi * (sq(radius) - sq(ra))));
    vparticles.resize(N + offset);
    fill(vparticles);
  }

  void fill(std::vector<Particle> &vparticles)
  {
    // // #pragma omp parallel for
    for (int p = 0; p < N; p++)
    {
      auto &particle = vparticles[offset + p];
      double r = radialDist(randEng);
      double t = thetaDist(randEng);
      particle.pos = Vector{r * std::cos(t), r * std::sin(t)} + pos;
    }
  }
  bool collisionDetection(Vector ppos)
  {
    double d = (ppos - pos).norm();
    return d <= radius;
  }
  void collide(Particle &particle, double dt)
  {
    particle.pos -= 0.5 * dt * particle.vel;
    particle.vel = -particle.vel;
    particle.pos = pos + radius * (particle.pos - pos).normalized() +
                   0.5 * dt * particle.vel;
  }
};

class SemiCircleGeometry
{
private:
  std::mt19937_64 &randEng;
  std::uniform_real_distribution<double> radialDist;
  std::uniform_real_distribution<double> thetaDist;
  std::uniform_real_distribution<double> raDist;
  std::uniform_real_distribution<double> aDist;
  std::normal_distribution<> &gauss;

  int surfVP = 0;
  int bottomVP = 0;

public:
  Vector pos;
  double radius;
  int offset;
  int N = 0;

  SemiCircleGeometry(Vector pos, double r, std::vector<Particle> &vparticles,
                     int rho, double a, std::mt19937_64 &gen,
                     std::normal_distribution<> &gauss)
      : randEng(gen), gauss(gauss), pos(pos), radius(r)
  {
    double ra = radius - std::sqrt(2) * a;
    radialDist = std::uniform_real_distribution<double>(ra, radius);
    thetaDist = std::uniform_real_distribution<double>(0, Pi);
    raDist = std::uniform_real_distribution<double>(-ra, ra);
    aDist = std::uniform_real_distribution<double>(0, a);
    offset = vparticles.size();
    surfVP =
        static_cast<int>(std::round(0.5 * rho * Pi * (sq(radius) - sq(ra))));
    bottomVP = static_cast<int>(std::round(rho * 2 * ra * a));
    N = surfVP + bottomVP;
    vparticles.resize(N + offset);
    fill(vparticles);
  }

  void fill(std::vector<Particle> &vparticles)
  {
    // #pragma omp parallel for
    for (int p = 0; p < surfVP; p++)
    {
      auto &particle = vparticles[offset + p];
      double r = radialDist(randEng);
      double t = thetaDist(randEng);
      particle.pos = Vector{r * std::cos(t), r * std::sin(t)} + pos;
      particle.vel = {gauss(randEng), gauss(randEng)};
      particle.rvel = {gauss(randEng), gauss(randEng)};
    }
    // #pragma omp parallel for
    for (int p = 0; p < bottomVP; p++)
    {
      auto &particle = vparticles[offset + p + surfVP];
      double x = raDist(randEng);
      double y = aDist(randEng);
      particle.pos = Vector{x, y} + pos;
      particle.vel = {gauss(randEng), gauss(randEng)};
      particle.rvel = {gauss(randEng), gauss(randEng)};
    }
  }
  bool collisionDetection(Vector ppos)
  {
    if (ppos[1] > pos[1])
    {
      double d = (ppos - pos).norm();
      return d <= radius;
    }
    return false;
  }
  void collide(Particle &particle, double dt)
  {
    double dy = std::abs(particle.pos[1] - pos[1]);
    Vector surf = pos + radius * (particle.pos - pos).normalized();
    double dr = (surf - particle.pos).norm();
    particle.pos -= 0.5 * dt * particle.vel;
    particle.vel = -particle.vel;
    if (dy > dr)
    {
      particle.pos = surf + 0.5 * dt * particle.vel;
    }
    else
    {
      particle.pos[1] = pos[1];
      particle.pos += 0.5 * dt * particle.vel;
    }
  }
};

//----------------------//

void writeParticles(int step, std::vector<Particle> &particles,
                    std::ofstream &os)
{
  int N = particles.size();
  std::string str1 = "";
  for (int i = 0; i < N; ++i)
  {
    str1 += std::to_string(step) + " " + std::to_string(i + 1) + " " +
            std::to_string(particles[i].pos[0]) + " " +
            std::to_string(particles[i].pos[1]) + "\n";
  }
  os << str1;
}

Tensor calculateMoi(Vector pos)
{
  double x = pos[0];
  double y = pos[1];
  Tensor moi({{y * y, -x * y, 0.0},
              {-x * y, x * x, 0.0},
              {0.0, 0.0, x * x + y * y}});

  return moi;
}

Vector shadowCircle(Vector &pos, Vector &center, double radius,
                    std::array<double, 2> box, std::array<int, 2> walls)
{
  Vector rpos = center;
  for (int d = 0; d < 2; d++)
  {
    if (walls[d] == 0)
    {
      if (center[d] - radius + box[d] < pos[d])
      {
        rpos[d] += box[d];
      }
      else if (center[d] + radius - box[d] > pos[d])
      {
        rpos[d] -= box[d];
      }
    }
  }
  return rpos;
}

// Simulation

Vector vf(double beta, double B1, Vector rs, Vector e)
{
  double ers = e.dot(rs);
  return B1 * (1 + beta * ers) * (ers * rs - e);
}

int main()
{
  int dim = 2;
  std::ofstream outdata;
  std::ofstream particlesFile;
  std::ofstream vparticlesFile;
  std::ofstream vebFile;
  outdata.open("output.txt");
  particlesFile.open("particles.txt");
  vparticlesFile.open("vparticles.txt");
  vebFile.open("veb.txt");
  outdata << "Step x y vx vy MSD \n";
  particlesFile << "Step Id x y\n";
  vparticlesFile << "Step Id x y\n";

  double veb = 0;

  double T = 1;
  double m = 1;
  std::array<double, 2> box = {100, 100};
  double a = 1;
  double dt = 0.1;
  int rho = 10;
  int steps = 150000;
  int thermo = 1;
  int trajthermo = 1;
  bool vp = true;
  double radius = 8.0;
  double B1 = 0.1;
  double beta = 0.0;
  std::array<int, 2> walls{0, 1};
  
  // Input file
  Vector spos = {15, 30};

  int obtsize = 20;
  double ox = 30;
  double oy = 0;

  std::ifstream intputFile("input.txt");
  std::string line;
  while (std::getline(intputFile, line))
  {
    std::istringstream ss(line);

    std::string value;
    ss >> value;

    if (value == "lx")
    {
      ss >> box[0];
    }
    else if (value == "ly")
    {
      ss >> box[1];
    }
    else if (value == "rho")
    {
      ss >> rho;
    }
    else if (value == "dt")
    {
      ss >> dt;
    }
    else if (value == "steps")
    {
      ss >> steps;
    }
    else if (value == "T")
    {
      ss >> T;
    }
    else if (value == "a")
    {
      ss >> a;
    }
    else if (value == "thermo")
    {
      ss >> thermo;
    }
    else if (value == "virtual")
    {
      ss >> vp;
    }
    else if (value == "traj")
    {
      ss >> trajthermo;
    }
    else if (value == "radius")
    {
      ss >> radius;
    }
    else if (value == "sx")
    {
      ss >> spos[0];
    }
    else if (value == "sy")
    {
      ss >> spos[1];
    }
    else if (value == "beta")
    {
      ss >> beta;
    }
    else if (value == "B1")
    {
      ss >> B1;
    }
    else if (value == "os")
    {
      ss >> obtsize;
    }
    else if (value == "ox")
    {
      ss >> ox;
    }
    else if (value == "oy")
    {
      ss >> oy;
    }
  }

  intputFile.close();

  int N = rho * box[0] * box[1];
  size_t nx = box[0] / a + walls[0];
  size_t ny = box[1] / a + walls[1];

  std::vector<Particle> particles(N, Particle());
  std::vector<std::vector<Vector>> vcell(nx,
                                         std::vector<Vector>(ny, {0.0, 0.0}));
  std::vector<std::vector<Vector>> vrcell(nx,
                                          std::vector<Vector>(ny, {0.0, 0.0}));
  std::vector<std::vector<int>> ncell(nx, std::vector<int>(ny, 0));

  std::random_device r{};
  std::mt19937_64 gen{r()};
  std::normal_distribution<> gauss(0, std::sqrt(T / m));
  std::uniform_real_distribution<> pos_x(0, box[0]);
  std::uniform_real_distribution<> pos_y(0, box[1]);
  std::uniform_real_distribution<> ran_shift(-a / 2, a / 2);
  std::uniform_real_distribution<> insidecell(0, a);

  // Sphere or circle
  // Vector spos = {box[0] / 2, box[1] / 2};
  Vector svel = {0.0, 0.0};
  Vector se = {1.0, 0.0};
  double somega = 0.0;

  double mass = N * Pi * radius * radius / (box[0] * box[1]);
  double moi = 0.5 * mass * radius * radius;

  std::uniform_real_distribution<> radial((radius - std::sqrt(2) * a), radius);
  std::uniform_real_distribution<> theta(0, 2 * Pi);
  int nf = static_cast<int>(std::floor(
      rho * Pi * (sq(radius) - sq(radius - std::sqrt(2) * a)) / sq(a)));
  int nwp = static_cast<int>(std::floor(rho * box[1] / a));
  std::vector<Particle> vparticles(nf + nwp, Particle());

  Vector cvel = {0.0, 0.0};
  SemiCircleGeometry circle(Vector{ox, oy}, obtsize, vparticles, rho, a, gen, gauss);

  for (auto &particle : particles)
  {
    particle.vel = {gauss(gen), gauss(gen)};
    cvel += particle.vel;
    double dist;
    bool cirCol = false;
    do
    {
      particle.pos[0] = pos_x(gen);
      particle.pos[1] = pos_y(gen);
      dist = (particle.pos - spos).norm();
      cirCol = circle.collisionDetection(particle.pos);
    } while (cirCol || (dist <= radius));
  }
  cvel /= N;
  for (auto &particle : particles)
  {
    particle.vel -= cvel;
    particle.cell = {static_cast<int>(std::floor(particle.pos[0] / a)),
                     static_cast<int>(std::floor(particle.pos[1] / a))};
  }

  writeParticles(-1, particles, particlesFile);
  cvel = spos;

  // Timing
  auto start = std::chrono::steady_clock::now();

  std::cout << "Number of particles " << N << "\n";
  std::cout << "Timesteps " << steps << "\n";
  std::cout << "dt " << dt << "\n";
  std::cout << "Box size " << box[0] << " " << box[1] << "\n";
  std::cout << "Squirmer : B1 " << B1 << " beta " << beta << "\n";
  std::cout
      << "<================================================================>"
      << "\n";
  std::cout << "Step    KE       Px          Py           L        Time"
            << "\n";

  // main loop
  for (int step = 0; step < steps; step++)
  {
    // reseting

    for (int i = 0; i < nx; i++)
    {
      for (int j = 0; j < ny; j++)
      {
        vcell[i][j] = {0.0, 0.0};
        vrcell[i][j] = {0.0, 0.0};
        ncell[i][j] = 0;
      }
    }

    // streaming

    for (int i=0; i<100; ++i){
    spos += svel * dt;
    se += Vector{-somega * se[1] * dt, somega * se[0] * dt};
    }

    for (auto circ : circles)
    {
      double csdiff = (spos - circ.pos).norm();
      // Semicircle squirmer collision
      if ((spos[1] >= circ.pos[1]) && csdiff <= (radius + circ.radius))
      {
        double tc = (radius + circ.radius - csdiff) / svel.norm();
        spos -= tc * svel;
        svel *= -1;
        spos += tc * svel;
      }
      else
      {
        Vector Veca = Vector{-circ.radius, 0} + circ.pos;
        Vector Vecb = Vector{circ.radius, 0} + circ.pos;
        double lineD = minDistLinePoint(Veca, Vecb, spos);
        if (lineD <= radius)
        {
          double tc = (radius - lineD) / svel.norm();
          spos -= tc * svel;
          Vector ab = Veca - Vecb;
          Vector nab = Vector{-ab[1], ab[0]}.normalized();
          svel -= 2 * svel.dot(nab) * nab;
          spos += tc * svel;
        }
      }
    }
    for (auto circ : upcircles)
    {
      double csdiff = (spos - circ.pos).norm();
      if ((spos[1] <= circ.pos[1]) && csdiff <= (radius + circ.radius))
      {
        double tc = (radius + circ.radius - csdiff) / svel.norm();
        spos -= tc * svel;
        svel *= -1;
        spos += tc * svel;
      }
      else
      {
        Vector Veca = Vector{-circ.radius, 0} + circ.pos;
        Vector Vecb = Vector{circ.radius, 0} + circ.pos;
        double lineD = minDistLinePoint(Veca, Vecb, spos);
        if (lineD <= radius)
        {
          double tc = (radius - lineD) / svel.norm();
          spos -= tc * svel;
          Vector ab = Veca - Vecb;
          Vector nab = Vector{-ab[1], ab[0]}.normalized();
          svel -= 2 * svel.dot(nab) * nab;
          spos += tc * svel;
        }
      }
    }

    // wall interaction
    for (size_t d = 0; d < dim; d++)
    {
      if (walls[d] == 1)
      {
        if (spos[d] + radius >= box[d])
        {
          svel[d] = -svel[d];
          spos[d] = 2 * (box[d] - radius) - spos[d];
        }
        else if (spos[d] - radius <= 0)
        {
          spos[d] = 2 * radius - spos[d];
          svel[d] = -svel[d];
        }
      }
      else if (walls[d] == 0)
      {
        if (spos[d] >= box[d])
        {
          spos[d] -= box[d];
        }
        else if (spos[d] < 0)
        {
          spos[d] += box[d];
        }
      }
    }

    Vector mom = {0.0, 0.0};
    double lmom = 0.0;

    // fluid particles
    // #pragma omp parallel
    {
      Vector privateMom = {0.0, 0.0};
      double privateLmom = 0.0;
      // #pragma omp for
      for (int p = 0; p < N; p++)
      {
        auto &particle = particles[p];
        particle.pos += particle.vel * dt;
        // periodic
        for (size_t d = 0; d < dim; d++)
        {
          if (particle.pos[d] >= box[d])
          {
            if (walls[d] == 1)
            {
              particle.vel = -particle.vel;
              particle.pos[d] = 2 * box[d] - particle.pos[d];
            }
            else
            {
              particle.pos[d] -= box[d];
            }
          }
          else if (particle.pos[d] < 0)
          {
            if (walls[d] == 1)
            {
              particle.vel = -particle.vel;
              particle.pos[d] = -particle.pos[d];
            }
            else
            {
              particle.pos[d] += box[d];
            }
          }
        }
        // inside circle
        Vector center = shadowCircle(particle.pos, spos, radius, box, walls);
        double d = (particle.pos - center).norm();
        if (d < radius)
        {
          Vector dp = -particle.vel;
          center -= 0.5 * svel * dt;
          particle.pos -= 0.5 * dt * particle.vel;
          Vector rs = (particle.pos - center).normalized();
          Vector surf = center + radius * rs;
          Vector rR = surf - center;
          Vector omegar = {-somega * rR[1], somega * rR[0]};
          // std::cout << vf(beta, B1, rs, se) << "\n";
          particle.vel =
              -particle.vel + 2 * (svel + omegar + vf(beta, B1, rs, se));
          particle.pos = surf + 0.5 * dt * particle.vel;
          // center += 0.5 * svel * dt;
          dp += particle.vel;
          privateMom += dp;
          privateLmom += rR[0] * dp[1] - rR[1] * dp[0];
        }

        for (auto circ : circles)
        {
          if (circ.collisionDetection(particle.pos))
          {
            circ.collide(particle, dt);
          }
        }

        for (auto circ : upcircles)
        {
          if (circ.collisionDetection(particle.pos))
          {
            circ.collide(particle, dt);
          }
        }
        // periodic
        for (size_t d = 0; d < dim; d++)
        {
          if (particle.pos[d] >= box[d])
          {
            if (walls[d] == 1)
            {
              particle.vel = -particle.vel;
              particle.pos[d] = 2 * box[d] - particle.pos[d];
            }
            else
            {
              particle.pos[d] -= box[d];
            }
          }
          else if (particle.pos[d] < 0)
          {
            if (walls[d] == 1)
            {
              particle.vel = -particle.vel;
              particle.pos[d] = -particle.pos[d];
            }
            else
            {
              particle.pos[d] += box[d];
            }
          }
        }
      }
      // #pragma omp critical
      {
        mom += privateMom;
        lmom += privateLmom;
      }
    }
    svel -= mom / mass;
    somega -= lmom / moi;

    // binning
    Vector shift = {ran_shift(gen), ran_shift(gen)};
    for (size_t d = 0; d < dim; d++)
    {
      if (walls[d] == 1)
      {
        shift[d] = shift[d] > 0 ? -shift[d] : shift[d];
      }
    }

    // shift circle
    spos += shift;

    // #pragma omp parallel for
    for (int p = 0; p < N; p++)
    {
      auto &particle = particles[p];
      Vector shifted_pos = particle.pos + shift;
      Vector rcorr = {0.0, 0.0};
      for (size_t d = 0; d < dim; d++)
      {
        if (walls[d] == 0)
        {
          if (shifted_pos[d] >= box[d])
          {
            shifted_pos[d] -= box[d];
            rcorr[d] -= box[d];
          }
          else if (shifted_pos[d] < 0)
          {
            shifted_pos[d] += box[d];
            rcorr[d] += box[d];
          }
        }
      }

      int i = static_cast<int>(std::floor(shifted_pos[0] / a)) + walls[0];
      int j = static_cast<int>(std::floor(shifted_pos[1] / a)) + walls[1];

      particle.cell = {i, j};
      particle.rvel = {gauss(gen), gauss(gen)};
      particle.rpos = rcorr;
      vcell[i][j] += particle.vel;
      vrcell[i][j] += particle.rvel;
      ncell[i][j]++;
      rcom[i][j] += particle.pos + rcorr;
    }
    if (vp)
    {
      // #pragma omp parallel for
      for (int p = 0; p < nf; p++)
      {
        auto &particle = vparticles[p];
        double r = radial(gen);
        double t = theta(gen);
        particle.pos = Vector{r * std::cos(t), r * std::sin(t)} + spos;

        for (size_t d = 0; d < dim; d++)
        {
          if (particle.pos[d] >= box[d])
          {
            particle.pos[d] -= box[d];
          }
          else if (particle.pos[d] < 0)
          {
            particle.pos[d] += box[d];
          }
        }

        int i = static_cast<int>(std::floor(particle.pos[0] / a)) + walls[0];
        int j = static_cast<int>(std::floor(particle.pos[1] / a)) + walls[1];

        particle.cell = {i, j};
        Vector center = shadowCircle(particle.pos, spos, radius, box, walls);

        Vector rs = (particle.pos - center).normalized();
        Vector rR = radius * rs;
        Vector surf = rR + center;
        Vector omegar = {-somega * rR[1], somega * rR[0]};
        particle.rvel = {gauss(gen), gauss(gen)};
        particle.vel = Vector{gauss(gen), gauss(gen)} + svel + omegar +
                       vf(beta, B1, rs, se);
        particle.vel = particle.rvel + svel + omegar + vf(beta, B1, rs, se);
        vcell[i][j] += particle.vel;
        vrcell[i][j] += particle.rvel;
        rcom[i][j] += particle.pos;
        ncell[i][j]++;
      }

      for (int p = 0; p < nwp; p++)
      {
        auto &particle = vparticles[nf + p];
        double x = pos_x(gen);
        double y = insidecell(gen) - a - shift[1];
        if (y < -a)
        {
          y += box[1];
        }
        particle.pos = Vector{x, y};

        for (size_t d = 0; d < dim; d++)
        {
          if (walls[d] == 0)
          {
            if (particle.pos[d] >= box[d])
            {
              particle.pos[d] -= box[d];
            }
            else if (particle.pos[d] < 0)
            {
              particle.pos[d] += box[d];
            }
          }
        }
        int i = static_cast<int>(std::floor(particle.pos[0] / a)) + walls[0];
        int j = static_cast<int>(std::floor(particle.pos[1] / a)) + walls[1];
        particle.cell = {i, j};
        particle.pos -= shift;
        particle.vel = Vector{gauss(gen), gauss(gen)};
        vcell[i][j] += particle.vel;
        vrcell[i][j] += particle.rvel;
        rcom[i][j] += particle.pos;
        ncell[i][j]++;
      }

      for (auto circle : circles)
      {
        circle.fill(vparticles);
      }

      for (int p = nf + nwp; p < vparticles.size(); p++)
      {
        auto &particle = vparticles[p];
        particle.pos += shift;

        for (size_t d = 0; d < dim; d++)
        {
          if (walls[d] == 0)
          {
            if (particle.pos[d] >= box[d])
            {
              particle.pos[d] -= box[d];
            }
            else if (particle.pos[d] < 0)
            {
              particle.pos[d] += box[d];
            }
          }
        }

        int i = static_cast<int>(std::floor(particle.pos[0] / a)) + walls[0];
        int j = static_cast<int>(std::floor(particle.pos[1] / a)) + walls[1];
        particle.cell = {i, j};
        vcell[i][j] += particle.vel;
        vrcell[i][j] += particle.rvel;
        rcom[i][j] += particle.pos;
        ncell[i][j]++;
        particle.pos -= shift;
      }
    }
    for (int i = 0; i < nx; i++)
    {
      for (int j = 0; j < ny; j++)
      {
        if (ncell[i][j] == 0)
        {
          vcell[i][j] = {0.0, 0.0};
          vrcell[i][j] = {0.0, 0.0};
          rcom[i][j] = {i * a + 0.5 * a,
                        j * a + 0.5 * a};
        }
        else
        {
          vcell[i][j] /= ncell[i][j];
          vrcell[i][j] /= ncell[i][j];
          rcom[i][j] /= ncell[i][j];
        }
      }
    }

    for (auto &particle : particles)
    {
      int i = particle.cell[0];
      int j = particle.cell[1];
      Vector rc = rcom[i][j];
      particle.rpos += particle.pos - rc;
      Vector difV = particle.vel - particle.rvel;
      moicell[i][j] += calculateMoi(particle.rpos);
      deltaL[i][j] += Vector3{0.0, 0.0, particle.rpos[0] * difV[1] -
      particle.rpos[1] * difV[0]};
    }
    if (vp)
    {
      for (auto &particle : vparticles)
      {
        int i = particle.cell[0];
        int j = particle.cell[1];
        Vector rc = rcom[i][j];
        particle.rpos += particle.pos - rc;
        Vector difV = particle.vel - particle.rvel;
        moicell[i][j] += calculateMoi(particle.rpos);
        deltaL[i][j] += Vector3{0.0, 0.0, particle.rpos[0] * difV[1] -
        particle.rpos[1] * difV[0]};
      }
    }

    for (int i = 0; i < nx; i++)
    {
      for (int j = 0; j < ny; j++)
      {
        auto det = moicell[i][j].determinant();
        auto adj = moicell[i][j].adjoint();
        Tensor invmoi = moicell[i][j].inverse();
        // Tensor invmoi = adj / det;
        if (det < 1e-14)
        {
          invmoi = Tensor{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        }
        deltaL[i][j] = invmoi * deltaL[i][j];
      }
    }

    // fluid particles
    // #pragma omp parallel for
    for (int p = 0; p < N; p++)
    {
      auto &particle = particles[p];
      int i = particle.cell[0];
      int j = particle.cell[1];
      Vector lcor = {-deltaL[i][j][2] * particle.rpos[1], deltaL[i][j][2] *
      particle.rpos[0]};
      particle.vel = particle.rvel + vcell[i][j] - vrcell[i][j];
    }
    if (vp)
    {
      mom = {0.0, 0.0};
      lmom = 0.0;
      // virtual particles
      // // #pragma omp parallel for
      for (int p = 0; p < nf; p++)
      {
        auto &particle = vparticles[p];
        int i = particle.cell[0];
        int j = particle.cell[1];
        // if (insideCircle(particle.pos, spos, radius))
        {
          Vector dp = -particle.vel;
          Vector lcor = {-deltaL[i][j][2] * particle.rpos[1], deltaL[i][j][2]
          * particle.rpos[0]};
          Vector center = shadowCircle(particle.pos, spos, radius, box, walls);
          particle.vel = particle.rvel + vcell[i][j] - vrcell[i][j];
          dp += particle.vel;
          mom += dp;
          Vector rr = particle.pos - center;
          lmom += rr[0] * dp[1] - rr[1] * dp[0];
          particle.pos -= shift;
        }
      }

      svel += mom / mass;
      somega += lmom / moi;
    }
    spos -= shift;

    if (step % trajthermo == 0)
    {
      double ke = 0;
      double mx = 0.0;
      double my = 0.0;

      writeParticles(step, particles, particlesFile);
      // if (vp) {
      //   writeParticles(step, vparticles, vparticlesFile);
      // }

      double kes = 0.5 * svel.dot(svel) / mass;
      double mxs = svel[0] / mass;
      double mys = svel[1] / mass;

      double lz = 0.0;

      // #pragma omp parallel for reduction(+ \
                                   : ke, mx, my, lz)
      for (int p = 0; p < N; p++)
      {
        auto &particle = particles[p];
        ke += particle.vel.dot(particle.vel);
        mx += particle.vel[0];
        my += particle.vel[1];
        lz += particle.pos[0] * particle.vel[1] -
              particle.pos[1] * particle.vel[0];
      }
      auto end = std::chrono::steady_clock::now();
      std::cout << step * dt << "  " << 0.5 * ke / N + kes << "   "
                << mx / N + mxs << "  " << my / N + mys << "  "
                << lz / N + lmom / moi << " "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                       .count()
                << "ms \n";
      start = end;
    }

    if (step % thermo == 0)
    {
      Vector diff = spos - cvel;

      outdata << step * dt << " " << spos[0] << " " << spos[1] << " " << svel[0]
              << " " << svel[1] << " " << diff.norm();
      if (B1 > 0)
      {
        outdata << " " << se[0] << " " << se[1] << " " << se.dot(svel);
      }
      outdata << "\n";
    }
    vebFile << (svel[0] * se[0] + svel[1] * se[1]) / B1 << "\n";
  }
  outdata.close();
  particlesFile.close();
  vparticlesFile.close();
  return 0;
}
