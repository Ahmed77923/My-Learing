--1) Display all patients with gender and date of birth
SELECT name, gender, dob
FROM patients;
--2) List all doctors and their specializations
SELECT name, specialization
FROM doctors;

--3) Show all visits in the last 30 days
SELECT *
FROM visits
WHERE visit_date >= CURRENT_DATE - INTERVAL '30 days';

--4) All visits handled by a specific doctor (e.g. Doctor_5)
SELECT visits.*
FROM visits
JOIN doctors ON visits.doctor_id = doctors.doctor_id
WHERE doctors.name = 'Doctor_5';

--5) Count total number of patients
SELECT COUNT(*)
FROM patients;

--6) Patient name, doctor name, visit date for every visit
SELECT patients.name, doctors.name, visits.visit_date
FROM visits
JOIN patients ON visits.patient_id = patients.patient_id
JOIN doctors ON visits.doctor_id = doctors.doctor_id;

--7) List patients and doctors who treated them
SELECT DISTINCT patients.name, doctors.name
FROM visits
JOIN patients ON visits.patient_id = patients.patient_id
JOIN doctors ON visits.doctor_id = doctors.doctor_id;

--8) Visit details including treatment cost
SELECT visits.visit_id, patients.name, doctors.name, visits.visit_date, treatment.cost
FROM visits
JOIN patients ON visits.patient_id = patients.patient_id
JOIN doctors ON visits.doctor_id = doctors.doctor_id
LEFT JOIN treatment ON visits.visit_id = treatment.visit_id;

--9) Visits where treatment cost > 300
SELECT visits.visit_id, patients.name, doctors.name, treatment.cost
FROM visits
JOIN patients ON visits.patient_id = patients.patient_id
JOIN doctors ON visits.doctor_id = doctors.doctor_id
JOIN treatment ON visits.visit_id = treatment.visit_id
WHERE treatment.cost > 300;

--10) Patients treated by cardiology doctors
SELECT DISTINCT patients.name, doctors.name
FROM visits
JOIN patients ON visits.patient_id = patients.patient_id
JOIN doctors ON visits.doctor_id = doctors.doctor_id
WHERE doctors.specialization = 'Cardiology';

--11) Count visits per doctor
SELECT doctors.name, COUNT(visits.visit_id)
FROM doctors
LEFT JOIN visits ON doctors.doctor_id = visits.doctor_id
GROUP BY doctors.doctor_id, doctors.name;

--12) Doctors with more than 50 visits
SELECT doctors.name, COUNT(visits.visit_id)
FROM doctors
JOIN visits ON doctors.doctor_id = visits.doctor_id
GROUP BY doctors.doctor_id, doctors.name
HAVING COUNT(visits.visit_id) > 50;

--13) Doctor with highest visits
SELECT doctors.name, COUNT(visits.visit_id)
FROM doctors
JOIN visits ON doctors.doctor_id = visits.doctor_id
GROUP BY doctors.doctor_id, doctors.name
ORDER BY COUNT(visits.visit_id) DESC
LIMIT 1;

--14) Average number of visits per doctor
SELECT AVG(visits_count)
FROM (
    SELECT COUNT(visits.visit_id) AS visits_count
    FROM doctors
    LEFT JOIN visits ON doctors.doctor_id = visits.doctor_id
    GROUP BY doctors.doctor_id
) subquery;

--15) Rank doctors by patients treated (MySQL 8+)
SELECT doctors.name, COUNT(DISTINCT visits.patient_id),
       RANK() OVER (ORDER BY COUNT(DISTINCT visits.patient_id) DESC)
FROM doctors
JOIN visits ON doctors.doctor_id = visits.doctor_id
GROUP BY doctors.doctor_id, doctors.name
ORDER BY 3;


--16) Calculate total hospital revenue from treatments
SELECT 
    SUM(cost) AS total_hospital_revenue
FROM treatment;
--17) Calculate total revenue generated per doctor
SELECT 
    d.name AS doctor_name,
    SUM(t.cost) AS total_revenue
FROM doctors d
JOIN visits v ON d.doctor_id = v.doctor_id
JOIN treatment t ON v.visit_id = t.visit_id
GROUP BY d.name
ORDER BY total_revenue DESC;


--18) Display daily revenue for the hospital
SELECT 
    v.visit_date,
    SUM(t.cost) AS daily_revenue
FROM visits v
JOIN treatment t ON v.visit_id = t.visit_id
GROUP BY v.visit_date
ORDER BY v.visit_date;

--19) Show monthly revenue trends
SELECT 
    DATE_TRUNC('month', v.visit_date) AS month,
    SUM(t.cost) AS monthly_revenue
FROM visits v
JOIN treatment t ON v.visit_id = t.visit_id
GROUP BY month
ORDER BY month;



--20) Identify the most profitable doctor
SELECT 
    d.name AS doctor_name,
    SUM(t.cost) AS total_revenue
FROM doctors d
JOIN visits v ON d.doctor_id = v.doctor_id
JOIN treatment t ON v.visit_id = t.visit_id
GROUP BY d.name
ORDER BY total_revenue DESC
LIMIT 1;

--21) Find patients who have visited more than 3 times
SELECT 
    p.name,
    COUNT(v.visit_id) AS visit_count
FROM patients p
JOIN visits v ON p.patient_id = v.patient_id
GROUP BY p.name
HAVING COUNT(v.visit_id) >= 3;


-- 22)Identify repeat patients and number of visit
SELECT 
    p.name,
    COUNT(v.visit_id) AS number_of_visits
FROM patients p
JOIN visits v ON p.patient_id = v.patient_id
GROUP BY p.name
HAVING COUNT(v.visit_id) > 1
ORDER BY number_of_visits DESC;



-- 23) List patients who have never received treatment
SELECT 
    p.name
FROM patients p
LEFT JOIN visits v ON p.patient_id = v.patient_id
LEFT JOIN treatment t ON v.visit_id = t.visit_id
WHERE t.treatment_id IS NULL;



-- 24) Find the average treatment cost per patient
SELECT 
    p.name,
    AVG(t.cost) AS avg_treatment_cost
FROM patients p
JOIN visits v ON p.patient_id = v.patient_id
JOIN treatment t ON v.visit_id = t.visit_id
GROUP BY p.name;

-- 25) Identify the top 10 patients by total treatment cost
SELECT 
    p.name,
    SUM(t.cost) AS total_spent
FROM patients p
JOIN visits v ON p.patient_id = v.patient_id
JOIN treatment t ON v.visit_id = t.visit_id
GROUP BY p.name
ORDER BY total_spent DESC
LIMIT 10;


-- 26)
CREATE TABLE male_copy AS
SELECT * FROM patients
WHERE gender = 'M';

select * from male_copy



--  27 create tabel copy data for specialization
CREATE TABLE copy_data AS
SELECT 
    p.patient_id, 
    p.name AS patient_name, 
    d.name AS doctor_name, 
    d.specialization
FROM patients p
JOIN visits v ON p.patient_id = v.patient_id
JOIN doctors d ON v.doctor_id = d.doctor_id
WHERE d.specialization = 'Pediatrics';

select * from copy_data

--   Create Tables for university database (FOR EXAMTION)

    CREATE TABLE department ( 
        department_id INT PRIMARY KEY,
        department_name VARCHAR(100)
    );
    CREATE TABLE student (
        student_id INT PRIMARY KEY,
        name VARCHAR(100),
        email VARCHAR(100),
        department_id INT ,
        enrollment_year INT,
        FOREIGN KEY (department_id) REFERENCES department(department_id)
    );
    CREATE TABLE course (
        course_id INT PRIMARY KEY,
        course_name VARCHAR(100),
        credit INT,
        department_id INT,
        FOREIGN KEY (department_id) REFERENCES department(department_id)
    );
    CREATE TABLE enrollment (
        enrollment_id INT PRIMARY KEY,
        student_id INT,
        course_id INT,
        semester VARCHAR(20),
        year INT,
        FOREIGN KEY (student_id) REFERENCES student(student_id),
        FOREIGN KEY (course_id) REFERENCES course(course_id)
    );
    CREATE TABLE grades (
        grade_id INT PRIMARY KEY,
        enrollment_id INT,
        grade INT,
        FOREIGN KEY (enrollment_id) REFERENCES enrollment(enrollment_id)
    );


\copy department(department_id,department_name) FROM 'C:/Users/PC/Desktop/Data/fat/department.csv' DELIMITER ',' CSV HEADER;
\copy student(student_id,name,email,department_id,enrollment_year) FROM 'C:/Users/PC/Desktop/Data/fat/student.csv' DELIMITER ',' CSV HEADER;
\copy course(course_id,course_name,credit,department_id) FROM 'C:/Users/PC/Desktop/Data/fat/course.csv' DELIMITER ',' CSV HEADER;
\copy enrollment(enrollment_id,student_id,course_id,semester,year) FROM 'C:/Users/PC/Desktop/Data/fat/enrollment.csv' DELIMITER ',' CSV HEADER;
\copy grades(grade_id,enrollment_id,grade) FROM 'C:/Users/PC/Desktop/Data/fat/grade.csv' DELIMITER ',' CSV HEADER;




-- Q B 2
select d.department_name, count(s.student_id) as Total_student
from students s
left join departments d on s.department_id = d.department_id 
group by d.department_name

-- Q B 1
select s.name , d.department_name
from students s
join departments d on s.department_id = d.department_id
where d.department_id = 1
-- q g) 
select c.course_name, avg(g.grade) as avrage
from courses c
join enrollments e on e.course_id = c.course_id
join grades g on g.enrollment_id = e.enrollment_id
group by c.course_name
-- q f)
select * from students
-- Q d) write SQL Query to retrieve all students recods from database
select count(*)  AS totale_students
from students 
-- q e) write SQL query to display courses where the highest mark obtained
--is greater than 80 showing the coure name and higest mark.
select c.course_name, g.grade 
from courses c 
join enrollments e on e.course_id = c.course_id
join grades g on g.enrollment_id = e.enrollment_id
where g.grade > 80
order by g.grade desc
-- q g) 
select c.course_name, avg(g.grade) as avrage
from courses c
join enrollment e on e.course_id = c.course_id
join grades g on g.enrollment_id = e.enrollment_id
group by c.course_name