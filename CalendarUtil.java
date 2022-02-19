import java.text.SimpleDateFormat;
import java.util.*;

public class CalendarUtil {
    public static String getWeekOfYear() {
        Calendar calendar = Calendar.getInstance();
        calendar.setMinimalDaysInFirstWeek(4);
        calendar.setFirstDayOfWeek(Calendar.MONDAY);
        int year = calendar.get(Calendar.YEAR);
        int weekOfYear = calendar.get(Calendar.WEEK_OF_YEAR);
        return String.format("%d-%02d", year, weekOfYear);
    }

    public static String getMonthOfYear() {
        Calendar calendar = Calendar.getInstance();
        calendar.setFirstDayOfWeek(Calendar.MONDAY);
        int year = calendar.get(Calendar.YEAR);
        int month = calendar.get(Calendar.MONTH) + 1;
        return String.format("%d-%02d", year, month);
    }

    public static String getCurrentTime() {
        Date date = new Date(System.currentTimeMillis());
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        return sdf.format(date);
    }

    public static String getCurrentTimeMillis() {
        return Long.toString(System.currentTimeMillis());
    }

    public static String getFirstDay(String type) {
        Calendar calendar = getFirstDayCalendar(type);
        Date date = calendar.getTime();
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        return sdf.format(date);
    }

    public static String getFirstDayTimeMillis(String type) {
        Calendar calendar = getFirstDayCalendar(type);
        return Long.toString(calendar.getTimeInMillis());
    }

    public static Calendar getFirstDayCalendar(String type) {
        Calendar calendar = Calendar.getInstance();
        calendar.setMinimalDaysInFirstWeek(4);
        calendar.setFirstDayOfWeek(Calendar.MONDAY);

        if (type.equals("week")) {
            calendar.set(Calendar.DAY_OF_WEEK, Calendar.MONDAY);
        }
        if (type.equals("month")) {
            calendar.set(Calendar.DAY_OF_MONTH, 1);
        }
        calendar.set(Calendar.HOUR_OF_DAY, 0);
        calendar.set(Calendar.MINUTE, 0);
        calendar.set(Calendar.SECOND, 0);
        return calendar;
    }

    public static List<Map<String, String>> getLatestYearFirstLastDay() {
        List<Map<String, String>> result = new ArrayList<>();

        for (int i = 0; i >= -11; i--) {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
            Calendar calendar = Calendar.getInstance();
            Date date = new Date();
            calendar.setTime(date);
            calendar.add(Calendar.MONTH, i);
            Date theDate = calendar.getTime();

            //第一天
            GregorianCalendar gcLast = (GregorianCalendar) Calendar.getInstance();
            gcLast.setTime(theDate);
            gcLast.set(Calendar.DAY_OF_MONTH, 1);
            String firstDay = sdf.format(gcLast.getTime());
            firstDay += " 00:00:00";

            //最后一天
            calendar.add(Calendar.MONTH, 1); //加一个月
            calendar.set(Calendar.DATE, 1);  //设置为该月第一天
            calendar.add(Calendar.DATE, -1); //再减一天即为上个月最后一天
            String lastDay = sdf.format(calendar.getTime());
            lastDay += " 23:59:59";

            int year = calendar.get(Calendar.YEAR);
            int month = calendar.get(Calendar.MONTH) + 1;

            Map<String, String> map = new HashMap<>();

            map.put("firstDay", firstDay);
            map.put("lastDay", lastDay);
            map.put("month", String.format("%d-%02d", year, month));

            result.add(map);
        }

        return result;
    }
}
